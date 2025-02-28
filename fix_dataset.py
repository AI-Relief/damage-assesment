import os
import shutil

# Define paths for train dataset
train_path = "dataset/train"
damaged_folder = os.path.join(train_path, "damaged")
major_folder = os.path.join(train_path, "Major")
minor_folder = os.path.join(train_path, "Minor")

# Ensure target folders exist
os.makedirs(major_folder, exist_ok=True)
os.makedirs(minor_folder, exist_ok=True)

# Move images based on filename
for file_name in os.listdir(damaged_folder):
    src_path = os.path.join(damaged_folder, file_name)

    if "Major" in file_name:  # If filename contains "Major", move to Major folder
        dest_path = os.path.join(major_folder, file_name)
        shutil.move(src_path, dest_path)
        print(f"✅ Moved {file_name} from damaged/ to Major/")

    elif "Minor" in file_name:  # If filename contains "Minor", move to Minor folder
        dest_path = os.path.join(minor_folder, file_name)
        shutil.move(src_path, dest_path)
        print(f"✅ Moved {file_name} from damaged/ to Minor/")

print("\n✅ Train dataset reorganization completed successfully!")

# Repeat the same process for test dataset
test_path = "dataset/test"
damaged_folder_test = os.path.join(test_path, "damaged")
major_folder_test = os.path.join(test_path, "Major")
minor_folder_test = os.path.join(test_path, "Minor")

# Ensure target folders exist
os.makedirs(major_folder_test, exist_ok=True)
os.makedirs(minor_folder_test, exist_ok=True)

# Move images based on filename in test set
for file_name in os.listdir(damaged_folder_test):
    src_path = os.path.join(damaged_folder_test, file_name)

    if "Major" in file_name:  # If filename contains "Major", move to Major folder
        dest_path = os.path.join(major_folder_test, file_name)
        shutil.move(src_path, dest_path)
        print(f"✅ Moved {file_name} from damaged/ to Major/ (Test Set)")

    elif "Minor" in file_name:  # If filename contains "Minor", move to Minor folder
        dest_path = os.path.join(minor_folder_test, file_name)
        shutil.move(src_path, dest_path)
        print(f"✅ Moved {file_name} from damaged/ to Minor/ (Test Set)")

print("\n✅ Test dataset reorganization completed successfully!")
