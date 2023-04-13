import csv

def fit(data_path, split):
    training_data = []
    test_data = []
    with open(data_path, newline='') as data:

        reader = csv.reader(data)
        total_rows = sum(1 for row in reader)
        data.seek(0)
        rows_to_process = int(total_rows * (split / 100))

        for i, row in enumerate(reader, start=1):
            if i > rows_to_process:
                break              
                
            floats = [float(val) for val in row[:4]]
            string = row[4]
            training_data.append(floats + [string])
        for i, row in enumerate(reader, start=rows_to_process):
            floats = [float(val) for val in row[:4]]
            string = row[4]
            test_data.append(floats + [string]) 
        return training_data, test_data
    

# data_sets = fit("iris_data_random.csv", 10)
# print(f"---------------------------------------- Train: {data_sets[0]}")
# print(f"---------------------------------------- Test: {data_sets[1]}")


