import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import glob
import shutil
import random

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SOURCE_DIR = r'D:\Project\resized_images_224'
OUTPUT_DIR = r'D:\Project\augmented_images_balanced'
TARGET_COUNT = 2000  # Target number of images per folder
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ---------------------------------------------------------
# AUGMENTATION PIPELINE
# ---------------------------------------------------------
augmentor = A.Compose([
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, fill_value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.2,
        rotate_limit=0,
        p=0.5,
        border_mode=cv2.BORDER_CONSTANT,
        fill_value=0
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.GaussNoise(var_limit=(10, 50), p=0.5),
    ], p=0.3),
])

def run_augmentation():
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory not found: {SOURCE_DIR}")
        return

    # Get class folders (subdirectories)
    class_folders = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
    
    if not class_folders:
        print(f"No subdirectories found in {SOURCE_DIR}. Processing root as a single class...")
        class_folders = ['.']

    print(f"Found {len(class_folders)} classes. Targeting {TARGET_COUNT} images per class.")

    for class_name in class_folders:
        class_src_path = os.path.join(SOURCE_DIR, class_name)
        class_dst_path = os.path.join(OUTPUT_DIR, class_name)
        
        if not os.path.exists(class_dst_path):
            os.makedirs(class_dst_path)

        # Find images in this class
        image_paths = []
        for ext in IMAGE_EXTENSIONS:
            image_paths.extend(glob.glob(os.path.join(class_src_path, f'*{ext}')))
        
        num_existing = len(image_paths)
        if num_existing == 0:
            print(f"Skipping {class_name}: No images found.")
            continue

        print(f"\nProcessing class: {class_name} ({num_existing} existing images)")

        # 1. Copy original images to destination
        print(f" - Copying {num_existing} originals...")
        for img_path in image_paths:
            shutil.copy2(img_path, class_dst_path)

        # 2. Generate augmentations to reach TARGET_COUNT
        if num_existing < TARGET_COUNT:
            num_to_generate = TARGET_COUNT - num_existing
            print(f" - Generating {num_to_generate} augmentations...")
            
            for i in tqdm(range(num_to_generate), desc=f"Augmenting {class_name}"):
                # Randomly pick an original image to augment
                src_img_path = random.choice(image_paths)
                image = cv2.imread(src_img_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented = augmentor(image=image)['image']
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                # Save with unique name
                base_name = os.path.splitext(os.path.basename(src_img_path))[0]
                ext = os.path.splitext(src_img_path)[1]
                output_filename = f"{base_name}_aug_{i}{ext}"
                cv2.imwrite(os.path.join(class_dst_path, output_filename), augmented_bgr)
        else:
            print(f" - Class already has {num_existing} images. No augmentations needed.")

    print(f"\nSuccess! Balanced dataset saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_augmentation()

