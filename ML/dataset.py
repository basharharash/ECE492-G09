import os
import pandas as pd
import numpy as np
from glob import glob

def load_csv_file_windows(file_path, window_size=5):
    """
    Loads one CSV file and creates sliding windows from the bean_temp series.
    Returns:
      - X: array of shape (num_windows, window_size) of bean temperatures
      - y: array of shape (num_windows,) of gas_setting values (the value right after each window)
    If the file doesn't have enough rows, returns (None, None).
    """
    df = pd.read_csv(file_path)
    df = df.sort_values(by="Time")
    
    # Ensure there are enough rows for one window
    if len(df) < window_size + 1:
        return None, None
    
    bean_temp = df['bean_temp'].values
    gas_setting = df['gas_setting'].values
    
    X_list = []
    y_list = []
    # Create sliding windows: for each window of "window_size" seconds, the target is the next gas setting.
    for i in range(len(df) - window_size):
        X_window = bean_temp[i:i+window_size]
        y_target = gas_setting[i+window_size]
        X_list.append(X_window)
        y_list.append(y_target)
        
    return np.array(X_list), np.array(y_list)

def load_all_csv_files_separated(modified_dir='modified', window_size=5):
    """
    Walks through the modified directory structure and loads CSV files from training and test folders separately.
    Training folders: "Valid", "Invalid"
    Testing folders: "Valid_test", "Invalid_test"
    
    Returns:
      - X_train, y_train, X_test, y_test arrays.
    """
    # Define which folder names correspond to training and testing sets.
    train_folders = ["Valid", "Invalid"]
    test_folders = ["Valid_test", "Invalid_test"]
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    # Assume structure is: modified/Q#/folder_name/
    quarter_folders = [os.path.join(modified_dir, q) 
                       for q in os.listdir(modified_dir) 
                       if os.path.isdir(os.path.join(modified_dir, q))]
    print("Quarter folders found:", quarter_folders)
    
    for quarter in quarter_folders:
        print("Processing quarter folder:", quarter)
        # For each folder in the quarter folder
        for folder in os.listdir(quarter):
            folder_path = os.path.join(quarter, folder)
            if not os.path.isdir(folder_path):
                continue
            
            # Determine if this folder is for training or testing
            if folder in train_folders:
                target_lists = (X_train_list, y_train_list)
            elif folder in test_folders:
                target_lists = (X_test_list, y_test_list)
            else:
                # Skip any folders not in our defined lists
                print("Skipping folder (not in target lists):", folder_path)
                continue
            
            print("Processing folder:", folder_path)
            # Get all CSV files in this folder
            csv_files = sorted(glob(os.path.join(folder_path, "*.csv")))
            print("Found CSV files:", csv_files)
            for file in csv_files:
                print("Loading file:", file)
                X_w, y_w = load_csv_file_windows(file, window_size)
                if X_w is not None:
                    target_lists[0].append(X_w)
                    target_lists[1].append(y_w)
                else:
                    print("File skipped (not enough data):", file)
    
    # Concatenate the windows from each file into single arrays, if any data was loaded
    X_train = np.concatenate(X_train_list, axis=0) if X_train_list else None
    y_train = np.concatenate(y_train_list, axis=0) if y_train_list else None
    X_test  = np.concatenate(X_test_list, axis=0)  if X_test_list  else None
    y_test  = np.concatenate(y_test_list, axis=0)  if y_test_list  else None
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    window_size = 5  # Predict within 5 seconds
    # Update the path based on your folder structure.
    # Here, modified_dir is assumed to be in the ML folder inside ECE492-G09.
    modified_dir = os.path.join('ML', 'modified')
    print("Using modified directory:", modified_dir)
    
    X_train, y_train, X_test, y_test = load_all_csv_files_separated(modified_dir=modified_dir, window_size=window_size)
    
    if X_train is not None and y_train is not None:
        print("Training set shapes: X:", X_train.shape, "y:", y_train.shape)
    else:
        print("No training data loaded.")
    
    if X_test is not None and y_test is not None:
        print("Test set shapes: X:", X_test.shape, "y:", y_test.shape)
    else:
        print("No test data loaded.")
