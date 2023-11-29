import random, os

def split_and_save_folds(input_file_path, output_dir):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    

    total_lines = len(lines)
    fold_size = total_lines // 10  # Assuming you want 10 folds

    for fold_num in range(10):
        fold_start = fold_num * fold_size
        fold_end = (fold_num + 1) * fold_size

        fold_data = lines[fold_start:fold_end]

        # Ensure each fold starts with a random index from the original file
        random_index = random.randint(0, total_lines - fold_size)
        fold_data[0] = lines[random_index]

        # Split each fold into train, test, and valid sets
        train_size = int(0.7 * fold_size)
        test_size = int(0.2 * fold_size)
        valid_size = fold_size - train_size - test_size

        train_data = fold_data[:train_size]
        test_data = fold_data[train_size:train_size + test_size]
        valid_data = fold_data[train_size + test_size:]

        # Save train, test, and valid sets with appropriate names
        save_fold_data(train_data, os.path.join(output_dir, f"fold{fold_num + 1}_train.txt"))
        save_fold_data(test_data, os.path.join(output_dir, f"fold{fold_num + 1}_test.txt"))
        save_fold_data(valid_data, os.path.join(output_dir, f"fold{fold_num + 1}_valid.txt"))

        print(f"Fold {fold_num + 1} saved.")

def save_fold_data(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(data)

# Example usage
input_file_path = "all_data_pair.txt"
output_directory = "folds"
split_and_save_folds(input_file_path, output_directory)
