#!/usr/bin/env python3
"""
Script per diagnosticare problemi nel dataset YOLO
Esegui questo script nella cartella /Users/salvatore/Development/KazaamLab/Mastopessi2/data
"""

import os
import glob
from pathlib import Path

def diagnose_yolo_dataset(data_path="/Users/salvatore/Development/KazaamLab/Mastopessi2/data"):
    print("🔍 DIAGNOSI DATASET YOLO")
    print("=" * 50)
    
    # 1. Verifica struttura cartelle
    print("📁 STRUTTURA CARTELLE:")
    train_img_path = os.path.join(data_path, "images/train")
    val_img_path = os.path.join(data_path, "images/val")
    train_lbl_path = os.path.join(data_path, "labels/train")
    val_lbl_path = os.path.join(data_path, "labels/val")
    
    paths = {
        "Train Images": train_img_path,
        "Val Images": val_img_path,
        "Train Labels": train_lbl_path,
        "Val Labels": val_lbl_path
    }
    
    for name, path in paths.items():
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")
    
    # 2. Conta file
    print("\n📊 CONTEGGIO FILE:")
    
    # Estensioni immagini comuni
    img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    
    def count_files(directory, extensions):
        if not os.path.exists(directory):
            return 0
        total = 0
        for ext in extensions:
            total += len(glob.glob(os.path.join(directory, ext)))
            total += len(glob.glob(os.path.join(directory, ext.upper())))
        return total
    
    def count_txt_files(directory):
        if not os.path.exists(directory):
            return 0
        return len(glob.glob(os.path.join(directory, "*.txt")))
    
    train_imgs = count_files(train_img_path, img_extensions)
    val_imgs = count_files(val_img_path, img_extensions)
    train_lbls = count_txt_files(train_lbl_path)
    val_lbls = count_txt_files(val_lbl_path)
    
    print(f"Train - Immagini: {train_imgs}, Labels: {train_lbls}")
    print(f"Val - Immagini: {val_imgs}, Labels: {val_lbls}")
    print(f"Totale immagini: {train_imgs + val_imgs}")
    
    # 3. Verifica corrispondenza file
    print("\n🔗 CORRISPONDENZA FILE:")
    
    def check_correspondence(img_path, lbl_path, split_name):
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"❌ {split_name}: Cartelle mancanti")
            return []
        
        # Get image files
        img_files = []
        for ext in img_extensions:
            img_files.extend(glob.glob(os.path.join(img_path, ext)))
            img_files.extend(glob.glob(os.path.join(img_path, ext.upper())))
        
        # Get label files
        lbl_files = glob.glob(os.path.join(lbl_path, "*.txt"))
        
        # Extract basenames
        img_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in img_files])
        lbl_basenames = set([os.path.splitext(os.path.basename(f))[0] for f in lbl_files])
        
        missing_labels = img_basenames - lbl_basenames
        orphan_labels = lbl_basenames - img_basenames
        
        print(f"{split_name}:")
        if missing_labels:
            print(f"  ❌ Immagini senza labels: {len(missing_labels)}")
            for missing in list(missing_labels)[:5]:  # Show first 5
                print(f"    - {missing}")
        if orphan_labels:
            print(f"  ⚠️  Labels senza immagini: {len(orphan_labels)}")
            for orphan in list(orphan_labels)[:5]:  # Show first 5
                print(f"    - {orphan}")
        if not missing_labels and not orphan_labels:
            print(f"  ✅ Tutti i file corrispondono")
        
        return lbl_files
    
    train_label_files = check_correspondence(train_img_path, train_lbl_path, "TRAIN")
    val_label_files = check_correspondence(val_img_path, val_lbl_path, "VAL")
    
    # 4. Verifica formato annotations
    print("\n📝 VERIFICA FORMATO ANNOTATIONS:")
    
    def check_annotations(label_files, split_name):
        if not label_files:
            print(f"❌ {split_name}: Nessun file di label trovato")
            return
        
        total_objects = 0
        problematic_files = []
        class_counts = {}
        
        for lbl_file in label_files[:10]:  # Check first 10 files
            try:
                with open(lbl_file, 'r') as f:
                    lines = f.readlines()
                    
                if len(lines) == 0:
                    problematic_files.append((lbl_file, "File vuoto"))
                    continue
                
                for line_num, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        problematic_files.append((lbl_file, f"Linea {line_num}: {len(parts)} parti invece di 5"))
                        continue
                    
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:5])
                        
                        # Check class range
                        if class_id < 0 or class_id > 7:  # 8 classes (0-7)
                            problematic_files.append((lbl_file, f"Linea {line_num}: classe {class_id} fuori range (0-7)"))
                        
                        # Check coordinate range
                        if not all(0 <= coord <= 1 for coord in [x, y, w, h]):
                            problematic_files.append((lbl_file, f"Linea {line_num}: coordinate fuori range [0,1]"))
                        
                        # Count classes
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
                        total_objects += 1
                        
                    except ValueError as e:
                        problematic_files.append((lbl_file, f"Linea {line_num}: errore conversione numeri"))
                        
            except Exception as e:
                problematic_files.append((lbl_file, f"Errore lettura file: {str(e)}"))
        
        print(f"{split_name}:")
        print(f"  📊 Oggetti totali trovati: {total_objects}")
        print(f"  📈 Distribuzione classi: {dict(sorted(class_counts.items()))}")
        
        if problematic_files:
            print(f"  ❌ File problematici: {len(problematic_files)}")
            for file_path, error in problematic_files[:5]:  # Show first 5
                filename = os.path.basename(file_path)
                print(f"    - {filename}: {error}")
        else:
            print(f"  ✅ Formato corretto")
    
    check_annotations(train_label_files, "TRAIN")
    check_annotations(val_label_files, "VAL")
    
    # 5. Suggerimenti
    print("\n💡 SUGGERIMENTI:")
    print("1. Assicurati che la struttura sia:")
    print("   data/")
    print("   ├── images/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   ├── labels/")
    print("   │   ├── train/")
    print("   │   └── val/")
    print("   └── data.yaml")
    print("\n2. Ogni immagine deve avere il suo file .txt corrispondente")
    print("3. Le classi devono essere nel range 0-7")
    print("4. Le coordinate devono essere normalizzate (0-1)")

if __name__ == "__main__":
    diagnose_yolo_dataset()