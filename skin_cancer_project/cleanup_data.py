import os
import shutil

def cleanup_splits(base_dir):
    splits = ['train', 'val', 'test']
    
    mapping = {
        'akiec_images': 'akiec',
        'bcc_images': 'bcc',
        'bkl_images': 'bkl',
        'df_images': 'df',
        'mel_images': 'melanoma',
        'nv_images': 'nevus',
        'vasc_images': 'vasc',
        'melanoma_images': 'melanoma', # Just in case
        'nevus_images': 'nevus',       # Just in case
    }

    for split in splits:
        split_path = os.path.join(base_dir, split)
        if not os.path.exists(split_path):
            print(f"Split {split} not found at {split_path}")
            continue
        
        print(f"Cleaning up {split} split...")
        
        # 1. Handle Project_Root
        project_root_path = os.path.join(split_path, 'Project_Root')
        if os.path.exists(project_root_path):
            print(f"Removing {project_root_path}")
            shutil.rmtree(project_root_path)

        # 2. Move images from _images folders to target folders
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if not os.path.isdir(folder_path):
                continue
            
            if folder in mapping:
                target_folder = mapping[folder]
                target_path = os.path.join(split_path, target_folder)
                
                if not os.path.exists(target_path):
                    os.makedirs(target_path)
                
                print(f"Moving contents from {folder} to {target_folder}")
                for item in os.listdir(folder_path):
                    s = os.path.join(folder_path, item)
                    d = os.path.join(target_path, item)
                    if os.path.exists(d):
                        if os.path.isdir(d):
                            shutil.rmtree(d)
                        else:
                            os.remove(d)
                    shutil.move(s, d)
                
                # Remove the now empty _images folder
                os.rmdir(folder_path)
        
        # 3. Final check: remove any empty folders that are NOT in our desired classes
        desired_classes = set(mapping.values())
        for folder in os.listdir(split_path):
            folder_path = os.path.join(split_path, folder)
            if os.path.isdir(folder_path):
                # If it's an empty directory and not a desired class, or even if it is a desired class but empty, 
                # we might want to know. But ImageFolder requires at least one file.
                if not os.listdir(folder_path):
                    print(f"Removing empty directory: {folder}")
                    os.rmdir(folder_path)

if __name__ == "__main__":
    cleanup_splits("d:/Project/skin_cancer_project/data/splits")
