# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: YOGARAJ .S
RegisterNumber:  212223040248
*/
```
```
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("/content/Employee_EX6.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data.drop(['Departments','left'],axis=1)
x.head()
y=data['left']
y.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![decision tree classifier model](sam.png)

DATA HEAD :


![Screenshot 2024-04-02 205452](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153482637/473644ec-2ea6-43b3-bcdf-9a03c73918df)

ACCURACY :

![Screenshot 2024-04-02 205510](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153482637/76bac2ef-0dc4-447c-9a46-6334e078e3be)

DATA PREDICT :

![Screenshot 2024-04-02 205532](https://github.com/yogaraj2/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/153482637/efb22feb-a40a-4f88-b8ac-3b84abf83f52)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
