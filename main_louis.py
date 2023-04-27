 # Importing required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_iris
import sklearn.metrics as metrics


# Loading datasets
iris = load_iris()

# Convert to pandas dataframe
iris_data = pd.DataFrame({
    'sepal length':iris.data[:,0],
    'sepal width':iris.data[:,1],
    'petal length':iris.data[:,2],
    'petal width':iris.data[:,3],
    'species':iris.target
})
iris_data.head()

# printing categories (setosa, versicolor, virginica)
print(iris.target_names)
# print flower features
print(iris.feature_names)

# setting independent (X) and dependent (Y) variables
X = iris_data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
Y = iris_data['species']  # Labels


# printing feature data
print(X[0:5])
# printing dependent variable values (0 = setosa, 1 = versicolor, 3 = virginica)
print(Y)

# splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 100)

# defining random forest classifier
clfr = RandomForestClassifier(random_state = 100)
clfr.fit(X_train, y_train)

# making prediction
Y_pred = clfr.predict(X_test)

# checking model accuracy
print("Accuracy:", metrics.accuracy_score(y_test, Y_pred))
cm = np.array(confusion_matrix(y_test, Y_pred))
print(cm)

# making predictions on new data
species_id = clfr.predict([[1, 5, 4, 6]])
iris.target_names[species_id]
print(iris.target_names[species_id])