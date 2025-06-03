# check_structure.py
import os
from pathlib import Path

def check_dataset_structure():
    """Verifica la struttura del dataset"""
    base_path = Path.cwd()
    
    print(f"Directory di lavoro corrente: {base_path}")
    print("\nStruttura delle cartelle:")
    
    # Percorsi da verificare
    paths_to_check = [
        'data',
        'data/raw_images',
        'data/raw_labels',
        'data/images',
        'data/images/train',
        'data/images/val',
        'data/labels',
        'data/labels/train',
        'data/labels/val',
        'data/dataset.yaml'
    ]
    
    for path in paths_to_check:
        full_path = base_path / path
        if full_path.exists():
            if full_path.is_dir():
                num_files = len(list(full_path.glob('*')))
                print(f"✓ {path} (contiene {num_files} file)")
            else:
                print(f"✓ {path} (file)")
        else:
            print(f"✗ {path} (non esiste)")
    
    # Verifica il contenuto di data.yaml se esiste
    yaml_path = base_path / 'data/data.yaml'
    if yaml_path.exists():
        print("\nContenuto di dataset.yaml:")
        with open(yaml_path, 'r') as f:
            print(f.read())

if __name__ == "__main__":
    check_dataset_structure()