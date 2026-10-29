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
BASE_DIR = r'D:\Project\skin_cancer_project\data\splits'
SOURCE_TRAIN_DIR = os.path.join(BASE_DIR, 'train')
OUTPUT_TRAIN_DIR = os.path.join(BASE_DIR, 'train_balanced')
TARGET_COUNT = 2000  # Target number of images per class
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# ---------------------------------------------------------
# AUGMENTATION PIPELINE
# ---------------------------------------------------------
augmentor = A.Compose([
    A.Rotate(limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(
        translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
        scale=(0.8, 1.2),
        rotate=(-20, 20),
        p=0.5,
        mode=cv2.BORDER_CONSTANT,
        cval=0
    ),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.5),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 3), p=0.5),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.5),
    ], p=0.3),
])

def run_balancing():
    if not os.path.exists(SOURCE_TRAIN_DIR):
        print(f"Error: Source training directory not found: {SOURCE_TRAIN_DIR}")
        return

    if not os.path.exists(OUTPUT_TRAIN_DIR):
        os.makedirs(OUTPUT_TRAIN_DIR)

    class_folders = [d for d in os.listdir(SOURCE_TRAIN_DIR) if os.path.isdir(os.path.join(SOURCE_TRAIN_DIR, d))]
    
    print(f"Found {len(class_folders)} classes in train split.")
    print(f"Targeting exactly {TARGET_COUNT} images per class.")

    for class_name in class_folders:
        class_src_path = os.path.join(SOURCE_TRAIN_DIR, class_name)
        class_dst_path = os.path.join(OUTPUT_TRAIN_DIR, class_name)
        
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

        # 1. If we have more than target, sample TARGET_COUNT
        if num_existing >= TARGET_COUNT:
            print(f" - Sampling {TARGET_COUNT} from {num_existing} originals...")
            sampled_paths = random.sample(image_paths, TARGET_COUNT)
            for img_path in tqdm(sampled_paths, desc=f"Copying {class_name}"):
                shutil.copy2(img_path, class_dst_path)
        
        # 2. If we have fewer than target, copy all and augment
        else:
            print(f" - Copying {num_existing} originals...")
            for img_path in image_paths:
                shutil.copy2(img_path, class_dst_path)
            
            num_to_generate = TARGET_COUNT - num_existing
            print(f" - Generating {num_to_generate} augmentations...")
            
            for i in tqdm(range(num_to_generate), desc=f"Augmenting {class_name}"):
                src_img_path = random.choice(image_paths)
                image = cv2.imread(src_img_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                augmented = augmentor(image=image)['image']
                augmented_bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
                
                base_name = os.path.splitext(os.path.basename(src_img_path))[0]
                ext = os.path.splitext(src_img_path)[1]
                output_filename = f"{base_name}_aug_{i}{ext}"
                cv2.imwrite(os.path.join(class_dst_path, output_filename), augmented_bgr)

    print(f"\nSuccess! Balanced training set saved in: {OUTPUT_TRAIN_DIR}")
    print("Next step: Replace 'train' with 'train_balanced' or update main.py.")

if __name__ == "__main__":
    run_balancing()
