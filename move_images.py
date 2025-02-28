import os
import shutil

# Define source folder where images are stored
source_folder = r"C:\Users\chris\OneDrive\Desktop\Student Project\Images\buildingDamage"

# Define target dataset paths
dataset_folder = r"C:\Users\chris\OneDrive\Desktop\Student Project\damage_assesment\dataset\train"

# Define categories and target paths
categories = {
    "Affected": os.path.join(dataset_folder, "Affected"),
    "Destroyed": os.path.join(dataset_folder, "Destroyed"),
    "Major": os.path.join(dataset_folder, "Major"),
    "Minor": os.path.join(dataset_folder, "Minor"),
    "NoDamage": os.path.join(dataset_folder, "NoDamage"),  # Ensure correct folder name
}

# Ensure target directories exist
for category_path in categories.values():
    os.makedirs(category_path, exist_ok=True)

# Debugging: Verify source folder exists
if not os.path.exists(source_folder):
    print(f"🚨 ERROR: Source folder does not exist: {source_folder}")
    exit()

# Get all files in the source folder
files = os.listdir(source_folder)

# Debugging: Show detected files before moving
print("\n📂 Checking source folder for images...\n")
if not files:
    print("🚨 ERROR: Source folder is empty! No images to move.")
    exit()

for file_name in files:
    print(f"   🔍 Found: [{file_name}] (Length: {len(file_name)})")

# Move files based on filename prefixes
files_moved = 0

for file_name in files:
    file_path = os.path.join(source_folder, file_name)

    if os.path.isfile(file_path):
        # Normalize filename (convert to lowercase, remove spaces, strip)
        file_name_cleaned = file_name.lower().replace(" ", "").strip()

        # Debugging: Show the cleaned filename
        print(f"🧐 Checking: [{file_name_cleaned}]")

        # Match the file to a category (more flexible search)
        target_path = None
        for category in categories.keys():
            if category.lower() in file_name_cleaned:
                target_path = os.path.join(categories[category], file_name)
                break

        if target_path is None:
            print(f"❌ Skipping {file_name} (No matching category)")
            continue

        # Debugging: Show move operation
        print(f"🔄 Moving {file_path} → {target_path}")

        try:
            shutil.move(file_path, target_path)
            print(f"✅ Successfully moved {file_name}")
            files_moved += 1
        except Exception as e:
            print(f"❌ ERROR moving {file_name}: {e}")

# Final confirmation
if files_moved == 0:
    print("\n🚨 No files were moved! Check file names and source path.")
else:
    print(f"\n✅ Successfully moved {files_moved} files!")
