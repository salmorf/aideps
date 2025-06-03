# fix_dataset_structure.py
import os
import shutil
from pathlib import Path
import random
import yaml

def fix_dataset_structure():
    """Sistema la struttura del dataset per YOLOv8"""
    base_path = Path.cwd()
    
    # 1. Verifica che esistano le immagini annotate
    raw_images = Path('data/raw_images')
    raw_labels = Path('data/raw_labels')
    
    if not raw_images.exists() or not raw_labels.exists():
        print("Errore: Assicurati di avere le cartelle data/raw_images e data/raw_labels")
        return
    
    # 2. Crea la struttura corretta
    print("Creazione struttura dataset...")
    data_path = Path('data')
    
    # Crea tutte le directory necessarie
    directories = [
        data_path / 'images' / 'train',
        data_path / 'images' / 'val',
        data_path / 'labels' / 'train',
        data_path / 'labels' / 'val'
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    
    # 3. Trova tutte le immagini con annotazioni
    annotated_pairs = []
    for img_path in raw_images.glob('*.jpg'):
        label_path = raw_labels / f"{img_path.stem}.txt"
        if label_path.exists():
            annotated_pairs.append((img_path, label_path))
    
    if not annotated_pairs:
        print("Errore: Nessuna immagine annotata trovata!")
        return
    
    print(f"Trovate {len(annotated_pairs)} immagini annotate")
    
    # 4. Split 80/20 per train/val
    random.shuffle(annotated_pairs)
    split_idx = int(len(annotated_pairs) * 0.8)
    
    train_pairs = annotated_pairs[:split_idx]
    val_pairs = annotated_pairs[split_idx:]
    
    # 5. Copia i file nelle directory corrette
    print("Copiando file...")
    for split, pairs in [('train', train_pairs), ('val', val_pairs)]:
        for img_path, label_path in pairs:
            # Copia immagine
            dest_img = data_path / 'images' / split / img_path.name
            shutil.copy2(img_path, dest_img)
            
            # Copia label
            dest_label = data_path / 'labels' / split / label_path.name
            shutil.copy2(label_path, dest_label)
    
    print(f"Dataset diviso: {len(train_pairs)} train, {len(val_pairs)} val")
    
    # 6. Crea dataset.yaml con percorsi ASSOLUTI
    dataset_config = {
        'path': str(data_path.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': {
            0: 'breast'
        },
        'nc': 1
    }
    
    yaml_path = data_path / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"\nDataset.yaml creato: {yaml_path}")
    print("\nContenuto di dataset.yaml:")
    with open(yaml_path, 'r') as f:
        print(f.read())
    
    # 7. Verifica finale
    print("\nVerifica finale:")
    for split in ['train', 'val']:
        img_count = len(list((data_path / 'images' / split).glob('*.jpg')))
        label_count = len(list((data_path / 'labels' / split).glob('*.txt')))
        print(f"{split}: {img_count} immagini, {label_count} labels")
    
    return str(yaml_path.absolute())

if __name__ == "__main__":
    yaml_path = fix_dataset_structure()
    if yaml_path:
        print(f"\nDataset pronto! Usa questo file per il training: {yaml_path}")