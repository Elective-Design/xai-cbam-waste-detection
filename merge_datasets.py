import os
import shutil
import yaml
from pathlib import Path
from tqdm import tqdm

# ====================================================================
# CONFIGURATION
# ====================================================================
# Unified Class List (Master Schema)
# 0: cardboard
# 1: glass
# 2: metal
# 3: paper
# 4: plastic
# 5: organic
# 6: trash
# 7: bottle
UNIFIED_CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic', 'trash', 'bottle']

# Mappings: {Original_Name: Master_Name}
# Note: Case sensitive based on data.yaml analysis
CLASS_MAPPING = {
    # Direct Matches
    'cardboard': 'cardboard',
    'glass': 'glass',
    'metal': 'metal',
    'paper': 'paper',
    'plastic': 'plastic',
    'organic': 'organic',
    'trash': 'trash',
    'bottle': 'bottle',
    
    # Variations
    'can': 'metal',
    'LDPE': 'plastic',
    'soft_plastic': 'plastic',
    'vinyl': 'plastic',
    'Almennt': 'trash',
    'Bottles': 'bottle',
    'Paper': 'paper',   # Case fix
    'Plastic': 'plastic' # Case fix
}

# Source Datasets (Absolute paths or relative to script)
# Based on ls -la output
SOURCE_DIRS = [
    "Trash classification.v2i.yolov11",
    "Trash-Waste Detection.v1i.yolov11",
    "Trash-ZeroWaste.v8-no-augment-dropped-trash.yolov11",
    "Trash.v6-online-new-old-mix.yolov11",
    "trash dataset.v1i.yolov11"
]

OUTPUT_DIR = "merged_dataset"

# ====================================================================
# UTILS
# ====================================================================
def read_dataset_yaml(dataset_path):
    """Reads the data.yaml of a dataset and returns names list."""
    yaml_path = Path(dataset_path) / "data.yaml"
    if not yaml_path.exists():
        print(f"Warning: {yaml_path} not found. Skipping dataset.")
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', [])

def create_dir_structure():
    """Creates train/val/test structure in output."""
    for split in ['train', 'valid', 'test']:
        for dtype in ['images', 'labels']:
            Path(OUTPUT_DIR, split, dtype).mkdir(parents=True, exist_ok=True)

def process_dataset(source_path, dataset_id):
    """Merges a single dataset into the output."""
    print(f"\nProcessing: {source_path}")
    source_path = Path(source_path)
    
    # 1. Get original class names
    original_names_raw = read_dataset_yaml(source_path)
    if not original_names_raw:
        return

    # Handle if names is list or dict
    if isinstance(original_names_raw, dict):
        original_names = [original_names_raw[i] for i in sorted(original_names_raw.keys())]
    else:
        original_names = original_names_raw

    # 2. Map original ID -> New ID
    # id_map = {old_id: new_id}
    id_map = {}
    for idx, name in enumerate(original_names):
        target_name = CLASS_MAPPING.get(name)
        if target_name and target_name in UNIFIED_CLASSES:
            id_map[idx] = UNIFIED_CLASSES.index(target_name)
        else:
            print(f"  Warning: Class '{name}' not found in mapping. Skipping annotations for this class.")
            id_map[idx] = -1 # Skip

    # 3. Process Splits
    # Note: Some datasets use 'valid', some 'val' usually. 
    # But standard yolo structure often has children folders as 'train', 'valid', 'test'
    # We check what exists.
    
    # Standard RoboFlow export usually has split folders at root level of dataset dir
    subfolders = {
        'train': 'train',
        'valid': 'valid',
        'test': 'test'
    }
    
    for split_name, target_split in subfolders.items():
        src_split_dir = source_path / split_name
        if not src_split_dir.exists():
            # Try 'val' instead of 'valid'
            if split_name == 'valid':
                src_split_dir = source_path / 'val'
            
            if not src_split_dir.exists():
                continue

        # Process Images and Labels
        images_dir = src_split_dir / 'images'
        labels_dir = src_split_dir / 'labels'
        
        if not images_dir.exists():
            continue
            
        print(f"  Merging split: {split_name}...")
        
        # Iterate over images
        for img_file in images_dir.glob('*.*'): # jpg, png, etc
            if img_file.name.startswith('.'): continue
            
            # Unique ID for file to prevent overwrite
            new_filename = f"d{dataset_id}_{img_file.name}"
            
            # Copy Image
            shutil.copy2(img_file, Path(OUTPUT_DIR, target_split, 'images', new_filename))
            
            # Process Label
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                new_label_lines = []
                with open(label_file, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        cls_id = int(parts[0])
                        
                        # Remap
                        new_id = id_map.get(cls_id, -1)
                        
                        if new_id != -1:
                            # Reconstruct line with new ID
                            new_line = f"{new_id} {' '.join(parts[1:])}\n"
                            new_label_lines.append(new_line)
                
                # Write new label file
                if new_label_lines:
                    with open(Path(OUTPUT_DIR, target_split, 'labels', f"d{dataset_id}_{img_file.stem}.txt"), 'w') as out_f:
                        out_f.writelines(new_label_lines)

# ====================================================================
# MAIN
# ====================================================================
if __name__ == "__main__":
    if Path(OUTPUT_DIR).exists():
        print("Removing existing merged_dataset...")
        shutil.rmtree(OUTPUT_DIR)
    
    create_dir_structure()
    
    for i, ds_path in enumerate(SOURCE_DIRS):
        process_dataset(ds_path, i)
        
    # Generate final data.yaml
    final_config = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'valid/images', # Usage valid/images is standard for Roboflow structure
        'test': 'test/images',
        'nc': len(UNIFIED_CLASSES),
        'names': UNIFIED_CLASSES
    }
    
    with open(Path(OUTPUT_DIR, 'data.yaml'), 'w') as f:
        yaml.dump(final_config, f, sort_keys=False)
        
    print("\nMerge Complete!")
    print(f"Unified Classes: {UNIFIED_CLASSES}")
    print(f"Data saved to: {os.path.abspath(OUTPUT_DIR)}")
