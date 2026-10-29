import os
import csv
import shutil

def move_images_from_csv(csv_path, source_folder, target_folder, column_name='image_name'):
    """
    Reads image names from a CSV and moves them from source_folder to target_folder.
    
    Args:
        csv_path (str): Path to the CSV file.
        source_folder (str): Directory where images are currently stored.
        target_folder (str): Directory where images should be moved.
        column_name (str): The header name in CSV containing image names.
    """
    
    # 1. Create target folder if it doesn't exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created target directory: {target_folder}")

    moved_count = 0
    not_found_count = 0

    # 2. Read the CSV file
    try:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                image_name = row.get(column_name)
                
                if not image_name:
                    continue
                
                # Construct full paths
                source_path = os.path.join(source_folder, image_name)
                dest_path = os.path.join(target_folder, image_name)

                # 3. Check if image exists (trying with and without extension)
                if not os.path.exists(source_path):
                    # Try common extensions if the exact name isn't found
                    for ext in ['.jpg', '.jpeg', '.png']:
                        if os.path.exists(source_path + ext):
                            source_path += ext
                            dest_path += ext
                            break

                if os.path.exists(source_path):
                    shutil.move(source_path, dest_path)
                    print(f"Moved: {os.path.basename(source_path)}")
                    moved_count += 1
                else:
                    print(f"Not Found: {image_name}")
                    not_found_count += 1

        print("\n--- Summary ---")
        print(f"Total images moved: {moved_count}")
        print(f"Total images not found: {not_found_count}")

    except FileNotFoundError:
        print(f"Error: The CSV file at {csv_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these paths to match your system
    CSV_FILE = r'D:\Project\vasc.csv'        # Added 'r' for raw string
    SOURCE_DIR = r'D:\Project\HAM10000_image'   # Added 'r'
    DEST_DIR = r'D:\Project\vasc_images'       # Added 'r' to fix \b issue
    COLUMN_WITH_NAMES = 'image_id'

    move_images_from_csv(CSV_FILE, SOURCE_DIR, DEST_DIR, COLUMN_WITH_NAMES)
