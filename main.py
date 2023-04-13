import numpy as np
from data_import import fit
import csv

def learn(data_path, percentage):
    result = []
    with open(data_path, newline='') as data:
        reader = csv.reader(data)
        total_rows = sum(1 for row in reader)
        data.seek(0)
        rows_to_process = int(total_rows * (percentage / 100))
        for i, row in enumerate(reader,start=1):
            if i > rows_to_process:
                print(f"processed {percentage}% of the data")
                break
            floats = [float(val) for val in row[:4]]
            string = row[4]
            result.append(floats + [string])
    return result

def algorithm(training_data, test_data):

    predictions = []
    for row_1 in test_data:
        a = np.array(row_1[0:4])
        nearest_neighbur = [100000, None]
        for row_2 in training_data:
            b = np.array(tuple(row_2[0:4]))
            dist = np.linalg.norm(a - b)
            if (dist < nearest_neighbur[0]):
                nearest_neighbur[0] = dist
                nearest_neighbur[1] = row_2[4]
        print(f"result: {nearest_neighbur[1]} | check: {row_1[4]}")
        if nearest_neighbur[1] != row_1[4]:
            print("mistake!")
        predictions.append(nearest_neighbur)

input = fit("iris_data_random.csv", 10)
algorithm(input[0], input[1])



