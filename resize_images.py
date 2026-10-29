import os
from PIL import Image

def resize_images_in_folders(source_folders, output_base_dir, target_size=(224, 224)):
    """
    Resizes images from multiple folders to a target size.
    
    Args:
        source_folders (list): List of paths to folders containing images.
        output_base_dir (str): Where to save the resized images.
        target_size (tuple): The desired (width, height).
    """
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    total_processed = 0

    for base_folder in source_folders:
        if not os.path.exists(base_folder):
            print(f"Warning: Folder not found: {base_folder}", flush=True)
            continue

        print(f"\nScanning: {base_folder}", flush=True)
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(base_folder):
            # 1. Skip the output directory and virtual environments
            if os.path.abspath(output_base_dir) in os.path.abspath(root) or '.venv' in root:
                continue

            # Filter for image files
            image_files = [f for f in files if f.lower().endswith(valid_extensions)]
            
            if not image_files:
                continue

            # 2. Determine target folder structure
            relative_path = os.path.relpath(root, base_folder)
            if relative_path == ".":
                target_folder = os.path.join(output_base_dir, "Project_Root")
            else:
                target_folder = os.path.join(output_base_dir, relative_path)

            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            print(f"--- Processing: {root} ({len(image_files)} images) ---", flush=True)

            for filename in image_files:
                try:
                    img_path = os.path.join(root, filename)
                    # Check if already exists to skip (optional)
                    save_path = os.path.join(target_folder, filename)
                    if os.path.exists(save_path):
                        continue

                    with Image.open(img_path) as img:
                        img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                        img_resized.save(save_path)
                    
                    total_processed += 1
                    if total_processed % 500 == 0:
                        print(f"Progress: {total_processed} images resized...", flush=True)
                
                except Exception as e:
                    print(f"Error processing {filename}: {e}", flush=True)

    print(f"\n--- Done! ---")
    print(f"Total images resized: {total_processed}")
    print(f"Resized images are saved in: {output_base_dir}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # 1. Add all folders you want to process here
    FOLDERS_TO_PROCESS = [
        r'D:\Project',  # This will now process all images directly in D:\Project
    ]

    # 2. Where the 224x224 images will be saved
    OUTPUT_DIRECTORY = r'D:\Project\resized_images_224'

    # 3. Target size
    SIZE = (224, 224)

    # Check if Pillow is installed
    try:
        import PIL
    except ImportError:
        print("Pillow library not found. Please install it using: pip install Pillow")
    else:
        resize_images_in_folders(FOLDERS_TO_PROCESS, OUTPUT_DIRECTORY, SIZE)
