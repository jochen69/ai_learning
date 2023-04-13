import random
import csv

def randomize_csv_rows(input_file, output_file):
    """
    Randomize the rows of a CSV file and save the result in a new CSV file.
    
    Args:
        input_file (str): Path to the input CSV file.
        output_file (str): Path to the output CSV file.
        
    Returns:
        None
    """
    # Read the input CSV file
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the header row
        rows = [row for row in reader]  # Read the remaining rows

    # Randomize the rows
    random.shuffle(rows)

    # Write the randomized rows to the output CSV file
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        writer.writerows(rows)  # Write the randomized rows

    print(f"Randomized rows written to {output_file} successfully.")

input_file = 'iris_data.csv'
output_file = 'iris_data_random.csv'

randomize_csv_rows(input_file, output_file)
