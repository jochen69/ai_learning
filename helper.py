import csv
import random

def data_import(file_path, test_size):
    """
    Load iris flower data from a CSV file and split it into training and test data.
    
    Args:
        file_path (str): File path of the CSV file containing the iris flower data.
        test_size (float): Percentage of data to be used as test data (0.0 to 1.0).
    
    Returns:
        tuple: A tuple containing X_train, X_test, Y_train, Y_test.
    """
    X_train = []  # List to store X_train values
    X_test = []   # List to store X_test values
    Y_train = []  # List to store Y_train values
    Y_test = []   # List to store Y_test values
    
    # Read CSV file
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        data = list(reader)
        random.shuffle(data)  # Shuffle the data randomly

        # Calculate the number of rows for test data based on the test_size percentage
        num_test_rows = int(len(data) * test_size/100)

        for row in data:
            # Append numerical features (columns 0 to 3) to either X_train or X_test list
            x = list(map(float, row[:4]))
            if len(X_test) < num_test_rows:
                X_test.append(x)
            else:
                X_train.append(x)
            
            # Append flower name (last column) to either Y_train or Y_test list
            y = row[4]
            if len(Y_test) < num_test_rows:
                Y_test.append(y)
            else:
                Y_train.append(y)
    print(f"data successfully imported. test size: {test_size} %; test rows: {num_test_rows}")
    return X_train, Y_train, X_test, Y_test
