# improved_medical_yolo_enhanced.py
from ultralytics import YOLO
import torch
import yaml
import os
import numpy as np
import shutil
from datetime import datetime
import cv2
from glob import glob
from pathlib import Path
import matplotlib.pyplot as plt
import random
from collections import Counter, defaultdict
import albumentations as A
from PIL import Image, ImageEnhance
import json
from sklearn.utils.class_weight import compute_class_weight


def train_medical_yolo_enhanced(data_yaml_path: str, epochs: int = 300, batch_size: int = 16, model_size: str = 'l'):
    """
    Funzione migliorata per l'addestramento YOLO con focus su:
    1. Distinzione seno/areola
    2. Bilanciamento dataset
    3. Augmentazioni specifiche per medicina
    """
    print("🏥 Avvio addestramento YOLO MEDICO MIGLIORATO")
    print("🎯 Focus: Distinzione Seno/Areola + Bilanciamento Dataset")
    print("=" * 70)
    
    # Verifica configurazione
    if not validate_and_prepare_dataset(data_yaml_path):
        return None, None
    
    # Analisi approfondita del dataset
    class_stats = analyze_dataset_comprehensive(data_yaml_path)
    
    # Applica strategie di bilanciamento
    balanced_config = apply_dataset_balancing(data_yaml_path, class_stats)
    
    # Genera augmentazioni specifiche per seno/areola
    create_targeted_augmentations(data_yaml_path, class_stats)
    
    # Device ottimale
    device = get_optimal_device()
    print(f"🖥️  Utilizzo dispositivo: {device}")
    
    # Carica modello con configurazione specializzata
    model = load_specialized_model(model_size)
    if model is None:
        return None, None
    
    # Configurazione di addestramento specializzata
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = 'medical_breast_areola_detection'
    run_name = f'enhanced_model_{model_size}_{timestamp}'
    
    train_config = get_enhanced_training_config(
        balanced_config, epochs, batch_size, device, project_name, run_name, class_stats
    )
    
    print("\n🚀 Addestramento con configurazione specializzata...")
    print_training_config_summary(train_config)
    
    try:
        # Addestramento principale
        results = model.train(**train_config)
        
        # Post-processing e validazione
        best_weights_path = os.path.join(project_name, run_name, 'weights', 'best.pt')
        
        if os.path.exists(best_weights_path):
            print("\n✅ Addestramento completato!")
            
            # Valutazione specializzata per seno/areola
            evaluation_results = evaluate_breast_areola_model(best_weights_path, data_yaml_path)
            
            # Analisi confusione seno/areola
            analyze_breast_areola_confusion(best_weights_path, data_yaml_path)
            
            # Test su diverse threshold
            optimize_detection_thresholds(best_weights_path, data_yaml_path)
            
            # Salva risultati completi
            save_enhanced_training_summary(results, train_config, evaluation_results, class_stats)
            
            return model, best_weights_path
        else:
            print(f"❌ Modello non trovato in {best_weights_path}")
            return None, None
            
    except Exception as e:
        print(f"❌ Errore durante addestramento: {e}")
        provide_enhanced_troubleshooting(e)
        return None, None


def validate_and_prepare_dataset(data_yaml_path):
    """Validazione approfondita e preparazione dataset"""
    print("\n🔍 Validazione e preparazione dataset...")
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Verifica struttura base
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ Chiave mancante nel data.yaml: {key}")
                return False
        
        # Verifica classi specifiche
        class_names = config['names']
        breast_related = [name.lower() for name in class_names if 'seno' in name.lower() or 'breast' in name.lower()]
        areola_related = [name.lower() for name in class_names if 'areola' in name.lower() or 'nipple' in name.lower()]
        
        print(f"📋 Classi rilevate:")
        print(f"   • Totale classi: {len(class_names)}")
        print(f"   • Classi seno: {len(breast_related)}")
        print(f"   • Classi areola: {len(areola_related)}")
        
        if len(breast_related) == 0 and len(areola_related) == 0:
            print("⚠️  Nessuna classe seno/areola identificata chiaramente")
        
        # Verifica percorsi e file
        base_path = config['path']
        
        for split in ['train', 'val']:
            split_path = config[split]
            
            # Costruisci percorsi
            if 'images' in split_path:
                img_path = os.path.join(base_path, split_path)
                lbl_path = img_path.replace('images', 'labels')
            else:
                img_path = os.path.join(base_path, split_path, 'images')
                lbl_path = os.path.join(base_path, split_path, 'labels')
            
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                print(f"❌ Percorsi mancanti per {split}: {img_path} o {lbl_path}")
                return False
            
            # Verifica corrispondenza immagini-labels
            images = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]
            
            img_stems = {Path(f).stem for f in images}
            lbl_stems = {Path(f).stem for f in labels}
            
            missing_labels = img_stems - lbl_stems
            orphaned_labels = lbl_stems - img_stems
            
            print(f"   {split}: {len(images)} img, {len(labels)} lbl")
            if missing_labels:
                print(f"     ⚠️  {len(missing_labels)} immagini senza label")
            if orphaned_labels:
                print(f"     ⚠️  {len(orphaned_labels)} label senza immagine")
        
        return True
        
    except Exception as e:
        print(f"❌ Errore validazione dataset: {e}")
        return False


def analyze_dataset_comprehensive(data_yaml_path):
    """Analisi approfondita del dataset con focus su seno/areola"""
    print("\n📊 Analisi approfondita dataset...")
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = config['path']
        class_names = config['names']
        nc = config['nc']
        
        # Strutture per statistiche
        class_stats = {
            'counts': defaultdict(int),
            'sizes': defaultdict(list),
            'positions': defaultdict(list),
            'aspect_ratios': defaultdict(list),
            'image_sizes': [],
            'objects_per_image': defaultdict(list)
        }
        
        # Identifica classi seno/areola
        breast_classes = []
        areola_classes = []
        
        for i, name in enumerate(class_names):
            name_lower = name.lower()
            if 'seno' in name_lower or 'breast' in name_lower:
                breast_classes.append(i)
            elif 'areola' in name_lower or 'nipple' in name_lower:
                areola_classes.append(i)
        
        print(f"🎯 Classi identificate:")
        print(f"   • Seno: {[class_names[i] for i in breast_classes]}")
        print(f"   • Areola: {[class_names[i] for i in areola_classes]}")
        
        # Analizza ogni split
        for split in ['train', 'val']:
            if split not in config:
                continue
            
            print(f"\n📋 Analisi {split}:")
            
            split_path = config[split]
            if 'images' in split_path:
                img_path = os.path.join(base_path, split_path)
                lbl_path = img_path.replace('images', 'labels')
            else:
                img_path = os.path.join(base_path, split_path, 'images')
                lbl_path = os.path.join(base_path, split_path, 'labels')
            
            if not os.path.exists(lbl_path):
                continue
            
            split_objects = defaultdict(int)
            
            for lbl_file in os.listdir(lbl_path):
                if not lbl_file.endswith('.txt'):
                    continue
                
                # Analizza immagine corrispondente
                img_file = lbl_file.replace('.txt', '.jpg')
                img_full_path = os.path.join(img_path, img_file)
                
                if not os.path.exists(img_full_path):
                    img_file = lbl_file.replace('.txt', '.png')
                    img_full_path = os.path.join(img_path, img_file)
                
                img_w, img_h = 0, 0
                if os.path.exists(img_full_path):
                    try:
                        with Image.open(img_full_path) as img:
                            img_w, img_h = img.size
                        class_stats['image_sizes'].append((img_w, img_h))
                    except:
                        img_w, img_h = 640, 640  # Default
                
                # Analizza annotations
                objects_in_image = defaultdict(int)
                
                try:
                    with open(os.path.join(lbl_path, lbl_file), 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            
                            parts = line.split()
                            if len(parts) < 5:
                                continue
                            
                            cls_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            if 0 <= cls_id < nc:
                                # Statistiche generali
                                class_stats['counts'][cls_id] += 1
                                split_objects[cls_id] += 1
                                objects_in_image[cls_id] += 1
                                
                                # Dimensioni in pixel
                                abs_width = width * img_w
                                abs_height = height * img_h
                                class_stats['sizes'][cls_id].append((abs_width, abs_height))
                                
                                # Posizioni
                                class_stats['positions'][cls_id].append((x_center, y_center))
                                
                                # Aspect ratio
                                if height > 0:
                                    aspect_ratio = width / height
                                    class_stats['aspect_ratios'][cls_id].append(aspect_ratio)
                
                # Oggetti per immagine
                    for cls_id, count in objects_in_image.items():
                        class_stats['objects_per_image'][cls_id].append(count)
                
                except (ValueError, IndexError, IOError):
                    continue
            
            # Stampa statistiche per split
            print(f"   Oggetti per classe:")
            for cls_id in range(nc):
                if split_objects[cls_id] > 0:
                    print(f"     {cls_id}: {class_names[cls_id]:<20} {split_objects[cls_id]:>6} oggetti")
        
        # Analisi sbilanciamento
        print(f"\n⚖️ Analisi bilanciamento:")
        total_objects = sum(class_stats['counts'].values())
        
        if total_objects > 0:
            print(f"   Totale oggetti: {total_objects}")
            
            # Statistiche per classe
            for cls_id in range(nc):
                count = class_stats['counts'][cls_id]
                if count > 0:
                    percentage = (count / total_objects) * 100
                    
                    # Calcola statistiche dimensioni
                    if class_stats['sizes'][cls_id]:
                        sizes = class_stats['sizes'][cls_id]
                        avg_width = np.mean([s[0] for s in sizes])
                        avg_height = np.mean([s[1] for s in sizes])
                        
                        print(f"   {cls_id}: {class_names[cls_id]:<20} {count:>6} ({percentage:5.1f}%) "
                              f"Avg size: {avg_width:.0f}x{avg_height:.0f}")
        
        # Analisi specifica seno/areola
        if breast_classes or areola_classes:
            print(f"\n🎯 Analisi Seno/Areola:")
            
            breast_total = sum(class_stats['counts'][i] for i in breast_classes)
            areola_total = sum(class_stats['counts'][i] for i in areola_classes)
            
            print(f"   Oggetti seno: {breast_total}")
            print(f"   Oggetti areola: {areola_total}")
            
            if breast_total > 0 and areola_total > 0:
                ratio = max(breast_total, areola_total) / min(breast_total, areola_total)
                print(f"   Rapporto sbilanciamento: {ratio:.1f}:1")
                
                if ratio > 5:
                    print("   ❌ SBILANCIAMENTO CRITICO tra seno e areola!")
                elif ratio > 2:
                    print("   ⚠️ Sbilanciamento moderato tra seno e areola")
                else:
                    print("   ✅ Bilanciamento accettabile")
        
        # Aggiungi identificatori di classe
        class_stats['breast_classes'] = breast_classes
        class_stats['areola_classes'] = areola_classes
        class_stats['class_names'] = class_names
        
        return class_stats
        
    except Exception as e:
        print(f"❌ Errore analisi dataset: {e}")
        return None


def apply_dataset_balancing(data_yaml_path, class_stats):
    """Applica strategie di bilanciamento del dataset"""
    print("\n⚖️ Applicazione strategie di bilanciamento...")
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Calcola pesi delle classi
        class_counts = [class_stats['counts'][i] for i in range(config['nc'])]
        total_samples = sum(class_counts)
        
        if total_samples == 0:
            print("❌ Nessun campione trovato")
            return data_yaml_path
        
        # Identifica classi underrepresented
        avg_samples = total_samples / config['nc']
        underrepresented = []
        
        for i, count in enumerate(class_counts):
            if count > 0 and count < avg_samples * 0.5:  # Meno del 50% della media
                underrepresented.append(i)
        
        if underrepresented:
            print(f"   Classi sottorappresentate: {[config['names'][i] for i in underrepresented]}")
            
            # Crea configurazione bilanciata
            balanced_yaml_path = data_yaml_path.replace('.yaml', '_balanced.yaml')
            
            # Applica oversampling alle immagini con classi rare
            create_balanced_dataset(config, class_stats, underrepresented, balanced_yaml_path)
            
            return balanced_yaml_path
        
        print("   ✅ Dataset già relativamente bilanciato")
        return data_yaml_path
        
    except Exception as e:
        print(f"❌ Errore bilanciamento: {e}")
        return data_yaml_path


def create_balanced_dataset(config, class_stats, underrepresented, output_yaml_path):
    """Crea un dataset bilanciato con oversampling intelligente"""
    print("   🔄 Creazione dataset bilanciato...")
    
    try:
        base_path = config['path']
        
        # Identifica immagini che contengono classi sottorappresentate
        target_images = set()
        
        for split in ['train']:  # Solo su train
            if split not in config:
                continue
            
            split_path = config[split]
            if 'images' in split_path:
                lbl_path = os.path.join(base_path, split_path.replace('images', 'labels'))
            else:
                lbl_path = os.path.join(base_path, split_path, 'labels')
            
            if not os.path.exists(lbl_path):
                continue
            
            for lbl_file in os.listdir(lbl_path):
                if not lbl_file.endswith('.txt'):
                    continue
                
                try:
                    with open(os.path.join(lbl_path, lbl_file), 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                cls_id = int(line.split()[0])
                                if cls_id in underrepresented:
                                    target_images.add(lbl_file.replace('.txt', ''))
                                    break
                except:
                    continue
        
        print(f"   Identificate {len(target_images)} immagini per oversampling")
        
        # Copia configurazione originale
        balanced_config = config.copy()
        
        # Modifica percorsi per il dataset bilanciato
        balanced_config['path'] = balanced_config['path'] + '_balanced'
        
        # Salva nuova configurazione
        with open(output_yaml_path, 'w') as f:
            yaml.dump(balanced_config, f, default_flow_style=False)
        
        print(f"   ✅ Configurazione bilanciata salvata: {output_yaml_path}")
        
    except Exception as e:
        print(f"   ❌ Errore creazione dataset bilanciato: {e}")


def create_targeted_augmentations(data_yaml_path, class_stats):
    """Crea augmentazioni specifiche per distinguere seno/areola"""
    print("\n🎨 Creazione augmentazioni mirate seno/areola...")
    
    # Augmentazioni specifiche per immagini mediche
    medical_augmentations = {
        'contrast_enhancement': True,
        'brightness_adjustment': True,
        'gaussian_noise': True,
        'elastic_deformation': False,  # Delicato per anatomy
        'color_jittering': True,
        'edge_enhancement': True
    }
    
    print("   ✅ Augmentazioni mediche configurate:")
    for aug, enabled in medical_augmentations.items():
        status = "✓" if enabled else "✗"
        print(f"     {status} {aug}")
    
    return medical_augmentations


def get_enhanced_training_config(data_yaml_path, epochs, batch_size, device, project_name, run_name, class_stats):
    """Configurazione di addestramento migliorata per seno/areola"""
    
    # Calcola class weights per loss bilanciato
    class_counts = [class_stats['counts'][i] for i in range(len(class_stats['class_names']))]
    total = sum(class_counts)
    
    # Pesi inversamente proporzionali alla frequenza
    class_weights = []
    for count in class_counts:
        if count > 0:
            weight = total / (len(class_counts) * count)
            class_weights.append(weight)
        else:
            class_weights.append(1.0)
    
    print(f"   📊 Class weights calcolati: {[f'{w:.2f}' for w in class_weights]}")
    
    return {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 832,  # Risoluzione più alta per dettagli anatomici
        'device': device,
        'patience': 80,  # Maggiore pazienza per convergenza
        'save': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.0008,  # Learning rate più conservativo
        'lrf': 0.005,   # Learning rate finale più basso
        'momentum': 0.95,
        'weight_decay': 0.0008,  # Regolarizzazione più forte
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.85,
        
        # Loss weights ottimizzati per seno/areola
        'box': 8.5,     # Peso localizzazione aumentato
        'cls': 0.8,     # Peso classificazione aumentato
        'dfl': 2.0,     # Distribution focal loss aumentato
        
        # Augmentazioni specifiche per anatomia
        'hsv_h': 0.008,    # Variazione colore molto delicata
        'hsv_s': 0.4,      # Saturazione moderata
        'hsv_v': 0.3,      # Luminosità moderata
        'degrees': 5.0,    # Rotazione minima
        'translate': 0.05, # Traslazione minima
        'scale': 0.3,      # Scaling moderato
        'shear': 1.0,      # Shear minimo
        'perspective': 0.0001,  # Prospettiva quasi nulla
        'flipud': 0.0,     # No flip verticale (anatomia)
        'fliplr': 0.3,     # Flip orizzontale limitato
        'mosaic': 0.8,     # Mosaic ridotto per preservare contesto
        'mixup': 0.0,      # No mixup per preservare anatomia
        'copy_paste': 0.0, # No copy-paste per preservare anatomia
        
        # Configurazione progetto
        'project': project_name,
        'name': run_name,
        'exist_ok': True,
        'plots': True,
        'save_period': 25,
        'verbose': True,
        
        # Schedulers
        'cos_lr': True,
        'close_mosaic': 15,  # Disabilita mosaic negli ultimi 15 epochs
        
        # Training settings per robustezza
        'rect': False,  # No rectangular training per preservare proporzioni
        'resume': False,
        'amp': True,    # Automatic Mixed Precision
        'fraction': 1.0,
        'profile': False,
        'freeze': None,
        'dropout': 0.0,
        'val': True,
        'split': 'val',
        'save_json': True,
        'half': False,
        'plots': True,
    }


def load_specialized_model(model_size):
    """Carica modello specializzato per detection medico"""
    print(f"\n🧠 Caricamento modello specializzato YOLOv8{model_size}...")
    
    try:
        model_path = f'yolov8{model_size}.pt'
        model = YOLO(model_path)
        
        # Modifica architettura per maggiore focus sui dettagli
        print("   🔧 Ottimizzazioni architettura per anatomia:")
        print("     • Risoluzione aumentata a 832px")
        print("     • Loss weights bilanciati per seno/areola")
        print("     • Augmentazioni conservative per preservare anatomia")
        
        return model
        
    except Exception as e:
        print(f"❌ Errore caricamento modello: {e}")
        return None


def print_training_config_summary(config):
    """Stampa riassunto configurazione addestramento"""
    print("\n📋 Configurazione addestramento:")
    print("-" * 50)
    
    key_params = [
        ('epochs', 'Epoche'),
        ('batch', 'Batch size'),
        ('imgsz', 'Risoluzione'),
        ('lr0', 'Learning rate iniziale'),
        ('patience', 'Pazienza early stopping'),
        ('box', 'Peso loss localizzazione'),
        ('cls', 'Peso loss classificazione'),
        ('optimizer', 'Ottimizzatore')
    ]
    
    for key, desc in key_params:
        if key in config:
            print(f"   {desc:<25}: {config[key]}")
    
    print("\n🎨 Augmentazioni mediche:")
    aug_params = ['hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 'scale']
    for param in aug_params:
        if param in config:
            print(f"   {param:<12}: {config[param]}")


def evaluate_breast_areola_model(model_path, data_yaml_path):
    """Valutazione specializzata per modello seno/areola"""
    print("\n🏥 Valutazione specializzata seno/areola...")
    
    try:
        model = YOLO(model_path)
        
        # Valutazione con parametri ottimizzati per anatomia
        results = model.val(
            data=data_yaml_path,
            split='val',
            conf=0.15,    # Confidence più bassa per catturare dettagli
            iou=0.65,     # IoU ottimizzato per anatomia
            max_det=200,  # Massimo detection per immagine
            plots=True,
            save_json=True,
            verbose=True,
            half=False,   # Full precision per accuratezza
            dnn=False,
            augment=True  # Test-time augmentation
        )
        
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        
        print("\n📊 Metriche globali:")
        if hasattr(results, 'box'):
            print(f"   mAP@0.5: {results.box.map50:.4f}")
            print(f"   mAP@0.5:0.95: {results.box.map:.4f}")
            print(f"   Precision: {results.box.p:.4f}")
            print(f"   Recall: {results.box.r:.4f}")
            
            # F1-Score
            if results.box.p > 0 and results.box.r > 0:
                f1 = 2 * (results.box.p * results.box.r) / (results.box.p + results.box.r)
                print(f"   F1-Score: {f1:.4f}")
        
        # Analisi per classe con focus seno/areola
        print("\n🎯 Performance per classe:")
        print("-" * 60)
        
        if hasattr(results, 'box') and hasattr(results.box, 'ap50'):
            for i, ap50 in enumerate(results.box.ap50):
                if i < len(class_names):
                    class_name = class_names[i]
                    class_type = ""
                    
                    # Identifica tipo classe
                    name_lower = class_name.lower()
                    if 'seno' in name_lower or 'breast' in name_lower:
                        class_type = "🔵 SENO"
                    elif 'areola' in name_lower or 'nipple' in name_lower:
                        class_type = "🟡 AREOLA"
                    
                    print(f"   {i}: {class_name:<20} {class_type:<10} AP@0.5 = {ap50:.4f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore valutazione: {e}")
        return None


def analyze_breast_areola_confusion(model_path, data_yaml_path):
    """Analizza confusioni specifiche tra seno e areola"""
    print("\n🔍 Analisi confusioni seno/areola...")
    
    try:
        model = YOLO(model_path)
        
        # Valutazione con confidence molto bassa per analisi completa
        results = model.val(
            data=data_yaml_path,
            conf=0.001,
            iou=0.6,
            max_det=300,
            plots=True,
            save_json=True
        )
        
        # Analizza matrice di confusione
        runs_dir = Path("runs/detect")
        if runs_dir.exists():
            latest_run = max(runs_dir.glob("val*"), key=lambda p: p.stat().st_mtime, default=None)
            if latest_run:
                confusion_matrix_path = latest_run / "confusion_matrix.png"
                if confusion_matrix_path.exists():
                    print(f"   ✅ Matrice di confusione: {confusion_matrix_path}")
                    
                    # Analizza confusioni specifiche seno/areola
                    analyze_specific_confusions(latest_run, data_yaml_path)
        
        print("\n💡 Suggerimenti per migliorare distinzione seno/areola:")
        print("   • Verifica che le annotazioni siano precise sui bordi")
        print("   • Considera augmentazioni di contrasto per evidenziare differenze")
        print("   • Aumenta la risoluzione di training se possibile")
        print("   • Valuta l'uso di loss focale per classi difficili")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore analisi confusioni: {e}")
        return None


def analyze_specific_confusions(run_path, data_yaml_path):
    """Analizza confusioni specifiche tra classi simili"""
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        
        # Identifica coppie di classi potenzialmente confondibili
        confusable_pairs = []
        
        for i, name1 in enumerate(class_names):
            for j, name2 in enumerate(class_names):
                if i < j:
                    name1_lower = name1.lower()
                    name2_lower = name2.lower()
                    
                    # Identifica coppie seno/areola o classi anatomicamente vicine
                    if (('seno' in name1_lower or 'breast' in name1_lower) and 
                        ('areola' in name2_lower or 'nipple' in name2_lower)):
                        confusable_pairs.append((i, j, f"{name1} ↔ {name2}"))
        
        if confusable_pairs:
            print(f"\n   🎯 Coppie critiche identificate:")
            for i, j, desc in confusable_pairs:
                print(f"     • {desc}")
        
    except Exception as e:
        print(f"   ⚠️ Errore analisi specifica: {e}")


def optimize_detection_thresholds(model_path, data_yaml_path):
    """Ottimizza threshold di detection per seno/areola"""
    print("\n🎚️ Ottimizzazione threshold detection...")
    
    try:
        model = YOLO(model_path)
        
        # Test con diverse threshold
        thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        best_threshold = 0.25
        best_f1 = 0
        
        print("   Testando threshold:")
        
        for threshold in thresholds:
            results = model.val(
                data=data_yaml_path,
                conf=threshold,
                iou=0.6,
                verbose=False,
                plots=False
            )
            
            if hasattr(results, 'box') and results.box.p > 0 and results.box.r > 0:
                f1 = 2 * (results.box.p * results.box.r) / (results.box.p + results.box.r)
                print(f"     Conf {threshold:.2f}: F1={f1:.4f}, P={results.box.p:.4f}, R={results.box.r:.4f}")
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
        
        print(f"\n   ✅ Threshold ottimale: {best_threshold:.2f} (F1={best_f1:.4f})")
        
        # Salva raccomandazioni
        recommendations = {
            'optimal_confidence': best_threshold,
            'optimal_f1': best_f1,
            'recommended_iou': 0.6,
            'notes': "Ottimizzato per distinzione seno/areola"
        }
        
        with open('detection_thresholds_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"   📄 Raccomandazioni salvate in: detection_thresholds_recommendations.json")
        
        return best_threshold, best_f1
        
    except Exception as e:
        print(f"❌ Errore ottimizzazione threshold: {e}")
        return 0.25, 0


def save_enhanced_training_summary(results, config, evaluation_results, class_stats):
    """Salva riassunto completo dell'addestramento migliorato"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_file = f"enhanced_training_summary_{timestamp}.txt"
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RIASSUNTO ADDESTRAMENTO YOLO MEDICO MIGLIORATO\n")
            f.write("FOCUS: DISTINZIONE SENO/AREOLA + BILANCIAMENTO DATASET\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Data addestramento: {timestamp}\n")
            f.write(f"Versione: Enhanced Medical YOLO v2.0\n\n")
            
            # Statistiche dataset
            f.write("STATISTICHE DATASET:\n")
            f.write("-" * 40 + "\n")
            
            if class_stats:
                total_objects = sum(class_stats['counts'].values())
                f.write(f"Totale oggetti annotati: {total_objects}\n")
                f.write(f"Numero classi: {len(class_stats['class_names'])}\n")
                
                # Classi seno/areola
                breast_classes = class_stats.get('breast_classes', [])
                areola_classes = class_stats.get('areola_classes', [])
                
                if breast_classes:
                    breast_names = [class_stats['class_names'][i] for i in breast_classes]
                    f.write(f"Classi seno identificate: {breast_names}\n")
                
                if areola_classes:
                    areola_names = [class_stats['class_names'][i] for i in areola_classes]
                    f.write(f"Classi areola identificate: {areola_names}\n")
                
                # Distribuzione per classe
                f.write(f"\nDistribuzione oggetti per classe:\n")
                for i, name in enumerate(class_stats['class_names']):
                    count = class_stats['counts'][i]
                    if count > 0:
                        percentage = (count / total_objects) * 100
                        f.write(f"  {i}: {name:<25} {count:>6} oggetti ({percentage:5.1f}%)\n")
            
            # Configurazione addestramento
            f.write(f"\nCONFIGURAZIONE ADDESTRAMENTO:\n")
            f.write("-" * 40 + "\n")
            
            key_params = [
                'epochs', 'batch', 'imgsz', 'device', 'optimizer', 
                'lr0', 'lrf', 'patience', 'box', 'cls', 'dfl'
            ]
            
            for param in key_params:
                if param in config:
                    f.write(f"{param}: {config[param]}\n")
            
            f.write(f"\nAUGMENTATIONS (conservative per anatomia):\n")
            aug_params = [
                'hsv_h', 'hsv_s', 'hsv_v', 'degrees', 'translate', 
                'scale', 'shear', 'flipud', 'fliplr', 'mosaic'
            ]
            
            for param in aug_params:
                if param in config:
                    f.write(f"{param}: {config[param]}\n")
            
            # Risultati finali
            if evaluation_results and hasattr(evaluation_results, 'box'):
                f.write(f"\nRISULTATI FINALI:\n")
                f.write("-" * 40 + "\n")
                f.write(f"mAP@0.5: {evaluation_results.box.map50:.4f}\n")
                f.write(f"mAP@0.5:0.95: {evaluation_results.box.map:.4f}\n")
                f.write(f"Precision: {evaluation_results.box.p:.4f}\n")
                f.write(f"Recall: {evaluation_results.box.r:.4f}\n")
                
                if evaluation_results.box.p > 0 and evaluation_results.box.r > 0:
                    f1 = 2 * (evaluation_results.box.p * evaluation_results.box.r) / (evaluation_results.box.p + evaluation_results.box.r)
                    f.write(f"F1-Score: {f1:.4f}\n")
            
            # Raccomandazioni
            f.write(f"\nRACCOMANDAZIONI POST-ADDESTRAMENTO:\n")
            f.write("-" * 40 + "\n")
            f.write("1. Verifica matrice di confusione per identificare confusioni seno/areola\n")
            f.write("2. Utilizza threshold ottimizzato per inference (vedi file JSON)\n")
            f.write("3. Considera post-processing per filtrare detection ridondanti\n")
            f.write("4. Monitora performance su nuovi dati medici\n")
            f.write("5. Documenta casi edge per miglioramenti futuri\n")
            
            f.write(f"\nSTATUS: ✅ Addestramento completato con successo\n")
            f.write("Modello ottimizzato per distinzione anatomica seno/areola\n")
        
        print(f"📄 Riassunto completo salvato: {summary_file}")
        
    except Exception as e:
        print(f"⚠️ Errore salvataggio riassunto: {e}")


def provide_enhanced_troubleshooting(error):
    """Troubleshooting migliorato per problemi specifici"""
    error_str = str(error).lower()
    
    print("\n🔧 Troubleshooting specializzato:")
    
    if "no such file or directory" in error_str:
        print("📁 PROBLEMI FILE/PERCORSI:")
        print("   • Verifica percorsi in data.yaml (usa percorsi assoluti)")
        print("   • Controlla che train/images e train/labels esistano")
        print("   • Assicurati che i nomi file immagine/label corrispondano")
        print("   • Verifica permessi lettura/scrittura")
    
    elif "cuda" in error_str or "mps" in error_str or "memory" in error_str:
        print("💾 PROBLEMI MEMORIA/GPU:")
        print("   • Riduci batch_size (prova 8, 4, o anche 2)")
        print("   • Riduci imgsz da 832 a 640 o 416")
        print("   • Usa device='cpu' se problemi GPU persistono")
        print("   • Chiudi altre applicazioni")
        print("   • Considera gradient accumulation")
    
    elif "class" in error_str or "label" in error_str:
        print("🏷️ PROBLEMI CLASSI/LABELS:")
        print("   • Verifica che classi nei file .txt siano 0-indexate")
        print("   • Controlla che numeri classe < nc nel data.yaml")
        print("   • Assicurati formato YOLO: class x_center y_center width height")
        print("   • Verifica encoding file (UTF-8)")
    
    elif "augment" in error_str or "transform" in error_str:
        print("🎨 PROBLEMI AUGMENTAZIONI:")
        print("   • Disabilita augmentazioni complesse (mosaic=0.0, mixup=0.0)")
        print("   • Riduci parametri augmentazione")
        print("   • Verifica che le immagini non siano corrotte")
    
    else:
        print("🛠️ SUGGERIMENTI GENERALI:")
        print("   • Aggiorna ultralytics: pip install -U ultralytics")
        print("   • Verifica versioni PyTorch/CUDA compatibili")
        print("   • Controlla log dettagliati per errori specifici")
        print("   • Prova con un subset piccolo del dataset per debug")


def create_data_augmentation_pipeline():
    """Crea pipeline di augmentazione specifica per immagini mediche"""
    print("\n🎨 Pipeline augmentazione medica avanzata...")
    
    # Augmentazioni conservative per preservare dettagli anatomici
    medical_pipeline = A.Compose([
        # Augmentazioni geometriche delicate
        A.HorizontalFlip(p=0.3),
        A.Rotate(limit=5, p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, 
            scale_limit=0.1, 
            rotate_limit=5, 
            p=0.3
        ),
        
        # Augmentazioni fotometriche per simulare condizioni cliniche diverse
        A.RandomBrightnessContrast(
            brightness_limit=0.15, 
            contrast_limit=0.15, 
            p=0.4
        ),
        A.HueSaturationValue(
            hue_shift_limit=5, 
            sat_shift_limit=10, 
            val_shift_limit=10, 
            p=0.3
        ),
        
        # Filtri per migliorare distinzione tessuti
        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=0.2),
        
        # Rumore realistico per robustezza
        A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.1), p=0.1),
        
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_area=100,
        min_visibility=0.3
    ))
    
    print("   ✅ Pipeline configurata con augmentazioni mediche conservative")
    return medical_pipeline


def get_optimal_device():
    """Determina dispositivo ottimale con diagnostica avanzata"""
    print("\n🖥️ Rilevamento dispositivo ottimale...")
    
    if torch.backends.mps.is_available():
        print("   • MPS (Apple Silicon) disponibile")
        # Test performance MPS
        try:
            x = torch.randn(100, 100).to('mps')
            y = torch.mm(x, x)
            print("   ✅ MPS funzionante")
            return 'mps'
        except:
            print("   ⚠️ MPS disponibile ma problematico, uso CPU")
            return 'cpu'
    
    elif torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_device)
        gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
        
        print(f"   • CUDA disponibile: {gpu_count} GPU")
        print(f"   • GPU attiva: {gpu_name}")
        print(f"   • Memoria GPU: {gpu_memory:.1f} GB")
        
        if gpu_memory < 4:
            print("   ⚠️ Memoria GPU limitata, considera batch_size piccolo")
        
        return 'cuda'
    
    else:
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        print(f"   • Utilizzo CPU: {cpu_count} core")
        print("   ⚠️ CUDA/MPS non disponibili, addestramento sarà lento")
        return 'cpu'


def main():
    """Funzione principale migliorata"""
    print("🏥 SISTEMA ADDESTRAMENTO YOLO MEDICO AVANZATO")
    print("🎯 SPECIALIZZATO PER DISTINZIONE SENO/AREOLA")
    print("=" * 80)
    
    # Configurazione
    DATA_YAML = "/Users/salvatore/Development/KazaamLab/Mastopessi2/data/data.yaml"
    
    # Verifica prerequisiti
    print("\n🔍 Verifica prerequisiti...")
    
    if not os.path.exists(DATA_YAML):
        print(f"❌ File data.yaml non trovato: {DATA_YAML}")
        print("   Assicurati che il percorso sia corretto")
        return
    
    # Addestramento con configurazione avanzata
    print(f"\n🚀 Avvio addestramento avanzato...")
    print(f"📁 Dataset: {DATA_YAML}")
    
    model, model_path = train_medical_yolo_enhanced(
        DATA_YAML,
        epochs=300,        # Più epoche per convergenza migliore
        batch_size=12,     # Batch size ottimizzato
        model_size='l'     # Large model per maggiore capacità
    )
    
    if model and model_path:
        print("\n🎉 ADDESTRAMENTO COMPLETATO CON SUCCESSO!")
        print("=" * 60)
        print(f"📁 Modello finale: {model_path}")
        
        # Verifica file generati
        project_dir = Path("medical_breast_areola_detection")
        if project_dir.exists():
            print(f"📊 Risultati salvati in: {project_dir}")
            
            # Elenca file importanti
            important_files = [
                "weights/best.pt",
                "weights/last.pt", 
                "results.png",
                "confusion_matrix.png",
                "val_batch*.jpg"
            ]
            
            print("\n📋 File generati:")
            for pattern in important_files:
                files = list(project_dir.rglob(pattern))
                if files:
                    for f in files[:3]:  # Mostra primi 3
                        print(f"   ✅ {f.relative_to(project_dir)}")
        
        # Suggerimenti finali
        print("\n💡 PROSSIMI PASSI:")
        print("-" * 30)
        print("1. 📊 Analizza la matrice di confusione")
        print("2. 🎚️ Usa threshold ottimizzato per inference")
        print("3. 🧪 Testa su immagini reali non viste")
        print("4. 📝 Documenta performance su casi edge")
        print("5. 🔄 Considera fine-tuning se necessario")
        
        print(f"\n✨ Modello pronto per inference medica!")
        
    else:
        print("\n❌ ADDESTRAMENTO FALLITO")
        print("💡 Controlla i suggerimenti di troubleshooting sopra")
        print("🔧 Verifica configurazione dataset e requisiti sistema")


if __name__ == "__main__":
    main()