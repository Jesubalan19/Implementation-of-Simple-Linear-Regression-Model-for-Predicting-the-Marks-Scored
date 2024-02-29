# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Jesubalan A
RegisterNumber:  212223240060
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores (1).csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
## Dataset:

![Screenshot 2024-02-29 221603](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/11e77615-8f88-49b0-903f-2d479269b99d)

## Head Values:

![Screenshot 2024-02-29 221617](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/542dcb51-3cd6-49cb-a127-92bf922232dd)

## Tail Values:

![Screenshot 2024-02-29 221628](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/41ef1520-0a28-4322-b900-b9571d053b8b)

## X and Y Values:

![Screenshot 2024-02-29 221646](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/20303b68-d7ec-46e8-b922-f9b5cbc891f1)
## Prediction values X and Y:

![Screenshot 2024-02-29 221657](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/b4cc2779-f5a9-421e-bf6f-24089bc9b84f)
## MSE, MAE and RMSE:

![Screenshot 2024-02-29 221706](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/ed6b78fb-1e30-44a5-b7ac-8530c1432dc7)
## Training Set:

![Screenshot 2024-02-29 221724](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/338e9031-40a1-407f-81fd-d89801e1f939)
## Testing Set:

![Screenshot 2024-02-29 221740](https://github.com/Jesubalan19/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979294/74a4ba65-603c-4b3c-b10c-992a861a15f4)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
