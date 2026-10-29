import os
import shutil
import random
from tqdm import tqdm

def split_dataset(source_dir, train_dir, val_dir, test_dir, train_pc=0.70, val_pc=0.15, test_pc=0.15):
    """
    Splits images from class-named subfolders into train, val, and test sets.
    """
    if not os.path.exists(source_dir):
        print(f"Source directory {source_dir} does not exist.")
        return

    # Create destination directories if they don't exist
    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Get class folders
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(classes)} classes: {classes}")

    for class_name in classes:
        print(f"\nSplitting class: {class_name}")
        
        class_src_path = os.path.join(source_dir, class_name)
        
        # Create class-specific subfolders in train, val, test
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)
        
        for d in [class_train_path, class_val_path, class_test_path]:
            if not os.path.exists(d):
                os.makedirs(d)

        # Get all images in this class
        images = [f for f in os.listdir(class_src_path) if os.path.isfile(os.path.join(class_src_path, f))]
        random.shuffle(images)

        num_images = len(images)
        num_train = int(num_images * train_pc)
        num_val = int(num_images * val_pc)
        # Remaining goes to test to avoid rounding issues
        
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        print(f" - Total: {num_images}")
        print(f" - Train: {len(train_images)}")
        print(f" - Val:   {len(val_images)}")
        print(f" - Test:  {len(test_images)}")

        # Copy files
        for img in tqdm(train_images, desc=f"  Copying {class_name} Train"):
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(class_train_path, img))
            
        for img in tqdm(val_images, desc=f"  Copying {class_name} Val"):
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(class_val_path, img))
            
        for img in tqdm(test_images, desc=f"  Copying {class_name} Test"):
            shutil.copy2(os.path.join(class_src_path, img), os.path.join(class_test_path, img))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split dataset into Train, Val, and Test sets.")
    parser.add_argument("--source", type=str, required=True, help="Path to the processed images directory.")
    parser.add_argument("--train_dir", type=str, default="data/splits/train", help="Path to save training set.")
    parser.add_argument("--val_dir", type=str, default="data/splits/val", help="Path to save validation set.")
    parser.add_argument("--test_dir", type=str, default="data/splits/test", help="Path to save test set.")
    parser.add_argument("--train_pc", type=float, default=0.70, help="Train percentage (0.0 to 1.0)")
    parser.add_argument("--val_pc", type=float, default=0.15, help="Val percentage (0.0 to 1.0)")
    parser.add_argument("--test_pc", type=float, default=0.15, help="Test percentage (0.0 to 1.0)")

    args = parser.parse_args()
    
    # Ensure percentages sum to 1.0
    total_pc = args.train_pc + args.val_pc + args.test_pc
    if not 0.99 <= total_pc <= 1.01:
        print(f"Warning: Percentages sum to {total_pc}, not 1.0. Normalizing...")
    
    split_dataset(
        args.source, 
        args.train_dir, 
        args.val_dir, 
        args.test_dir, 
        args.train_pc, 
        args.val_pc, 
        args.test_pc
    )
