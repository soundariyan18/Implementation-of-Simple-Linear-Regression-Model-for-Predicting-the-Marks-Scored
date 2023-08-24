# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages.

2.Display the values predicted using scatter plot and predict.

3.Plot the graph according to the given input.

4.End the program 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: M.N.SOUNDARIYAN
RegisterNumber: 212222230146 
*/
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()

df.tail()

x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

y_pred

y_test

plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,regressor.predict(x_test),color="red")
plt.title("Hours vs Scores (Test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print('MSE= ',mse)

mae=mean_absolute_error(y_test,y_pred)
print('MAE= ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```


## Output:

df.head()

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221433.png)

df.tail()

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221502.png)

Array values of x

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221529.png)

Array value of Y

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221600.png)

Values of Y prediction

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221714.png)

Array values of Y test

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221741.png)

Training Set Graph

![MODEL](https://github.com/soundariyan18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/blob/main/Screenshot%202023-08-24%20221814.png)

Test Set Graph

![MODEL]()
9. Values of MSE, MAE and RMSE






## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
