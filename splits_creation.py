import pandas as pd
import numpy as np

# Read the CSV file containing slide IDs
data = pd.read_csv('camelyon16.csv')

# Shuffle the rows randomly
data = data.sample(frac=1).reset_index(drop=True)

# Determine the number of rows in the dataset
num_rows = len(data)

# Determine the number of rows for train, val, and test
num_train = int(0.6 * num_rows)
num_val = int(0.2 * num_rows)
num_test = num_rows - num_train - num_val

# Create a new DataFrame for the dataset
dataset = pd.DataFrame(columns=['train', 'val', 'test'])

# Fill the train column with slide IDs
dataset['train'] = data['slide_id'].iloc[:num_train]

# Fill the val column with slide IDs
dataset['val'] = data['slide_id'].iloc[num_train:num_train+num_val]

# Fill the test column with slide IDs
dataset['test'] = data['slide_id'].iloc[num_train+num_val:]

# Fill any missing entries with NaN
dataset = dataset.fillna("")

# Save the dataset to a new CSV file named "split_dataset.csv"
dataset.to_csv('split_dataset.csv', index=False)

print("Dataset created and saved as split_dataset.csv.")
