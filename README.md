# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Use the standard libraries in python for finding linear regression.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn

4.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Adchayakiruthika M S
RegisterNumber: 212223230005

import pandas as pd
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

df=pd.read_csv('/content/Mammal_Cart.csv')
data=df.copy()
data.describe()

label_encoder=LabelEncoder()
data['Toothed']=label_encoder.fit_transform(data['Toothed'])
data['Hair']=label_encoder.fit_transform(data['Hair'])
data['Breathes']=label_encoder.fit_transform(data['Breathes'])
data['Legs']=label_encoder.fit_transform(data['Legs'])
data['Species']=label_encoder.fit_transform(data['Species'])

x=data.drop('Species',axis=1)
y=data['Species']

clf=DecisionTreeClassifier(criterion='gini')

clf.fit(x,y)
plt.figure(figsize=(18,6))
plot_tree(clf,feature_names=x.columns,class_names=['Reptile','Mammal'],filled=True)
plt.show()
```

## Output:
## Head:
![image](https://github.com/Adchayakiruthika18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147139995/248992ef-264d-4d0b-b086-2d6874eb1768)

![image](https://github.com/Adchayakiruthika18/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147139995/85cb7194-f21c-49c9-9231-78f7f265ee90)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

