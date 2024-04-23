import os
import shutil

# Existing data gathering for Ashton
directory_path_ashton = "Data"
files_list_ashton = os.listdir(directory_path_ashton)
files_set_ashton = set(files_list_ashton)
print(f"The current file count for Ashton is: {len(files_set_ashton)}")

# Existing data gathering for Claire
directory_path_claire = "Data2"
files_list_claire = os.listdir(directory_path_claire)
files_set_claire = set(files_list_claire)
print(f"The current file count for Claire is: {len(files_set_claire)}")

# Combine both sets
files_set_combined = files_set_ashton.union(files_set_claire)

estimated_total = len(files_set_claire) + len(files_set_ashton)
files_set_combined = files_set_ashton.union(files_set_claire)

# New directory for the combined files
output_directory = "CombinedFiles"
os.makedirs(output_directory, exist_ok=True)  # Create directory if it doesn't exist

print(f"The total file count is: {len(files_set_combined)}")
print(f"The number of removed files: {estimated_total - len(files_set_combined)}")

for file_name in files_set_combined:
    source_ashton = os.path.join(directory_path_ashton, file_name)
    source_claire = os.path.join(directory_path_claire, file_name)
    destination = os.path.join(output_directory, file_name)

    # Check which source the file is in and copy it
    if os.path.exists(source_ashton):
        shutil.copy(source_ashton, destination)
    elif os.path.exists(source_claire):
        shutil.copy(source_claire, destination)

# verification
directory_path_total = "CombinedFiles"
files_list_total = os.listdir(directory_path_total)
files_set_total = set(files_list_total)
print(f"The combination directory count is: {len(files_set_total)}")

