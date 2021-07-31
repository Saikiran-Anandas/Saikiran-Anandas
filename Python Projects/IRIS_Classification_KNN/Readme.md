# IRIS_CLASSIFICATION_KNN


Like ” Hello World” program in every programming language, Iris flower classification is the basic classification example of machine learning. It has only 150 datasets and contains the parameters like Sepal length, Sepal width, Petal length, Petal width which is used to classify iris flowers named as Iris Setosa , Iris Versicolor, Iris Virginica.

Download: Iris Flowers Dataset from My Project Repository..

Iris flower classification using KNN

This project focuses on the classification of iris flowers into their respective species by using the KNN machine-learning algorithm. The three species in this classification problem include setosa, versicolor, and virginica. The explanatory variables include sepal length, sepal width, pedal length, petal width. We are essentially trying to predict the species of the iris flower based on physical features!

The data consists of continuous numeric values which describe the dimensions of the respective features. We will be training the model based on these features. We will be using Python to understand and train our model. Numpy, Pandas and matplotlib are some of the inbuilt libraries in Python.

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
data = pd.read_csv('iris.csv')

```

First of all, all the required libraries like numpy and pandas are imported and iris datasets is loaded which was downloaded earlier

We visualize the top 5 data of the datasets.

```python
X = data.iloc[:,:4].values
y = data['species'].values
```

Here, we split the data into dependent and independent variables. Let us start by training our model with some of the samples. We will be using an inbuilt library called ‘train_test_split’ which divides our data set into a ratio of 80:20. It can be done by the following code:

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

Let us start building our model and predicting accuracy of every algorithm used. We can also check which gives the best result.

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model = KNeighborsClassifier()
model.fit(X_train,y_train)
predictions=model.predict(X_test)
print(accuracy_score(y_test,predictions))
print(model.predict(X_test))
```

accuracy= 0.93333333333333
['virginica' 'virginica' 'setosa' 'setosa' 'setosa' 'virginica'
 'versicolor' 'versicolor' 'virginica' 'virginica' 'versicolor'
 'virginica' 'setosa' 'setosa' 'setosa' 'setosa' 'virginica' 'versicolor'
 'setosa' 'versicolor' 'setosa' 'virginica' 'setosa' 'virginica'
 'virginica' 'versicolor' 'virginica' 'setosa' 'virginica' 'versicolor']
In Machine Learning, no any algorithm gives 100% accuracy. You can use other methods like SVM, Linear Regression, Random Forest etc. 

Stay updated for more tutorials . Bye for today !
