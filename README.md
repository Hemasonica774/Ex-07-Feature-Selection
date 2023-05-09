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
import pandas as pd
import numpy as np
df = pd.read_csv("titanic_dataset.csv")
df
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'].astype(str))
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[['Age']] = imputer.fit_transform(df[['Age']])
print("Feature selection")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X, y)
print(X_new)
df_new = pd.DataFrame(X_new, columns=['Pclass', 'Age', 'Fare'])
df_new['Survived'] = y.values
df_new.to_csv('titanic_transformed.csv', index=False)
print(df_new)


```

# OUTPUT

![image](https://user-images.githubusercontent.com/118361409/237005708-aa836d9c-b7f8-426c-b11c-822f87dc4efe.png)

![image](https://user-images.githubusercontent.com/118361409/237005751-9788217d-5229-4cc3-9c70-d8081a6a1f28.png)

![image](https://user-images.githubusercontent.com/118361409/237005920-bfa32e78-2420-476e-a827-cc7e8423da17.png)



# CODE

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("CarPrice.csv")
df
df.isnull().sum()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import ExtraTreesRegressor
df = df.drop(['car_ID', 'CarName'], axis=1)
le = LabelEncoder()
df['fueltype'] = le.fit_transform(df['fueltype'])
df['aspiration'] = le.fit_transform(df['aspiration'])
df['doornumber'] = le.fit_transform(df['doornumber'])
df['carbody'] = le.fit_transform(df['carbody'])
df['drivewheel'] = le.fit_transform(df['drivewheel'])
df['enginelocation'] = le.fit_transform(df['enginelocation'])
df['enginetype'] = le.fit_transform(df['enginetype'])
df['cylindernumber'] = le.fit_transform(df['cylindernumber'])
df['fuelsystem'] = le.fit_transform(df['fuelsystem'])
 df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=1)
print("Univariate Selection")
selector = SelectKBest(score_func=f_regression, k=10)
X_train_new = selector.fit_transform(X_train, y_train)
mask = selector.get_support()
selected_features = X_train.columns[mask]
model = ExtraTreesRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_
indices = np.argsort(importance)[::-1]
selected_features = X_train.columns[indices][:10]
df_new = pd.concat([X_train[selected_features], y_train], axis=1)
df_new.to_csv('CarPrice_new.csv', index=False)
print(df_new)X = df.iloc[:, :-1]
y =

```

 OUTPUT
 
![image](https://user-images.githubusercontent.com/118361409/237007096-500102c5-4d7d-4947-901e-84c9aa09b7f5.png)

![image](https://user-images.githubusercontent.com/118361409/237007259-d647b844-faa8-4925-8a6e-70deccb98ba1.png)

![image](https://user-images.githubusercontent.com/118361409/237007296-93aa6f3d-f19d-40f1-ac44-24439169ae84.png)



 # RESULT

The various feature selection techniques has been performed on a dataset and saved the data to a file.
