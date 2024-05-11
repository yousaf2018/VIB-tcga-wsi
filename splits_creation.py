import pandas as pd
from sklearn.model_selection import train_test_split

# Read the CSV file containing the case IDs and labels
df = pd.read_csv("camelyon16.csv")

# Separate the case IDs and labels
case_ids = df['case_id']
labels = df['label']
case_id_list = case_ids.values
train_list = case_id_list[:48]
val_list = case_id_list[48:59]
test_list= case_id_list[59:69]
# Create DataFrames for train, val, and test sets
train_df = pd.DataFrame({'train': train_list})
val_df = pd.DataFrame({'val': val_list})
test_df = pd.DataFrame({'test': test_list})

# Concatenate the DataFrames
split_df = pd.concat([train_df, val_df, test_df], axis=1)

# Save the DataFrame to a CSV file
split_df.to_csv("split_0.csv", index=False)

print("CSV file saved successfully.")
