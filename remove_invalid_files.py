import os
from PIL import Image, UnidentifiedImageError

# Path to your images folder
images_dir = "F:/vscode/python/task4/dataset/images"

# Log file to record removed files
log_file = "invalid_files_log.txt"

# Open log file for writing
with open(log_file, "w") as log:
    log.write("Invalid Files Log\n")
    log.write("=================\n")

# Iterate through all files in the directory
for filename in os.listdir(images_dir):
    file_path = os.path.join(images_dir, filename)
    try:
        # Try to open the image
        with Image.open(file_path) as img:
            img.verify()  # Verify that the file is a valid image
    except (UnidentifiedImageError, FileNotFoundError):
        # Log and remove invalid files
        with open(log_file, "a") as log:
            log.write(f"Invalid file removed: {file_path}\n")
        print(f"Removing invalid file: {file_path}")
        os.remove(file_path)  # Remove the invalid file

print("Invalid file removal complete. Check the log for details.")
