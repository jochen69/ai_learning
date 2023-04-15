import numpy as np
from helper import data_import

# Definition des Algorithmus

class KNearestNeighbor:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.Y_train = None
        print(f"initializing with K={k} neighbours:")

    def distance(self, x1, x2):
        x1 = np.array(x1)
        x2 = np.array(x2)
        distance = np.linalg.norm(x1 - x2)
        return distance
    
    def fit(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        print(f"algorithm trained")

    def predict(self, X_test):
        y_pred = []
        for x_test in X_test:
            dist = []
            for x_train in self.X_train:
                dist.append(self.distance(x_test, x_train))
            
            k_nearest_indices = sorted(range(len(dist)), key=lambda i: dist[i])[:self.k]
            k_nearest_labels = [self.Y_train[i] for i in k_nearest_indices]
            predicted_label = max(k_nearest_labels, key=k_nearest_labels.count)
            y_pred.append(predicted_label)
        return y_pred

# Daten importieren:
data = data_import("data/iris_data.csv", int(input("test size: ")))
X_train, Y_train, X_test, Y_test = data[:4]

# Modell erstellen und trainieren

knn = KNearestNeighbor(k=int(input("k: ")))
knn.fit(X_train, Y_train)

# Vorhersagen für Testdaten machen
y_pred = knn.predict(X_test)

# Vorhersagen auswerten
def evaluate():
    error_count = 0
    for i in range(len(Y_test)):
        if y_pred[i] != Y_test[i]:
            print(f"Error: {y_pred[i]} ({Y_test[i]})")
            error_count += 1
        else:
            print(f"Result: {y_pred[i]} ({Y_test[i]})")
    print(f"Errors: {error_count}")

evaluate()
