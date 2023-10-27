import json
from sklearn.model_selection import train_test_split

# Load your data from the input JSON file
with open("C:\programs\E2E-ECPE\Subtask_2_2_train.json", "r") as file:
    data = json.load(file)

# Define the number of folds
num_folds = 10

# Initialize variables to keep track of the fold data
fold_data = []

# Split the data into folds
for i in range(num_folds):
    train_ratio = 0.7  # 70% for training
    test_ratio = 0.15  # 15% for testing
    valid_ratio = 0.15  # 15% for validation

    # Split the data for the current fold
    train_data, temp_data = train_test_split(data, test_size=1 - train_ratio, random_state=42)
    test_data, valid_data = train_test_split(temp_data, test_size=valid_ratio / (valid_ratio + test_ratio), random_state=42)

    # Save the split datasets with specific file names for the current fold
    fold_train_file = f"fold{i+1}_train.json"
    fold_test_file = f"fold{i+1}_test.json"
    fold_valid_file = f"fold{i+1}_valid.json"

    with open(fold_train_file, "w") as train_file:
        json.dump(train_data, train_file, indent=4)

    with open(fold_test_file, "w") as test_file:
        json.dump(test_data, test_file, indent=4)

    with open(fold_valid_file, "w") as valid_file:
        json.dump(valid_data, valid_file, indent=4)

    # Store the file names for this fold
    fold_data.append({
        "fold": i + 1,
        "train_file": fold_train_file,
        "test_file": fold_test_file,
        "valid_file": fold_valid_file
    })

# Save the fold data to a JSON file
with open("fold_data.json", "w") as fold_data_file:
    json.dump(fold_data, fold_data_file, indent=4)
