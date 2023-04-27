import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Step 1: Load and preprocess the Iris flower dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the input data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 2: Initialize the neural network classifier
clf = MLPClassifier(hidden_layer_sizes=(8,), activation='relu', solver='adam', max_iter=1000, random_state=42)

# Step 3: Train the neural network
clf.fit(X_train, y_train)

# Step 4: Make predictions
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)

# Step 5: Compute accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Step 6: Print the accuracy
print(y_pred_test)
print('Training accuracy:', train_accuracy)
print('Testing accuracy:', test_accuracy)