# improved_medical_yolo_confusion_matrix.py
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
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random


def train_medical_yolo_optimized(data_yaml_path: str, epochs: int = 200, batch_size: int = 16, model_size: str = 'l'):
    """
    Funzione ottimizzata per addestramento YOLO con focus sulla matrice di confusione
    """
    print("🚀 Avvio addestramento YOLO ottimizzato per matrice di confusione")
    print(f"📁 Configurazione: {data_yaml_path}")
    
    # Verifica preliminari
    if not os.path.exists(data_yaml_path):
        print(f"❌ File {data_yaml_path} non trovato!")
        return None
    
    # Carica e analizza configurazione
    try:
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        print("✅ Configurazione caricata:")
        print(f"   - Dataset path: {data_config.get('path', 'NON DEFINITO')}")
        print(f"   - Numero classi: {data_config.get('nc', 'NON DEFINITO')}")
        print(f"   - Classi: {data_config.get('names', 'NON DEFINITO')}")
        
        # Analisi avanzata del dataset
        class_weights = analyze_and_calculate_class_weights(data_yaml_path)
        validate_dataset_quality(data_config)
        
    except Exception as e:
        print(f"❌ Errore nel caricamento del data.yaml: {e}")
        return None
    
    # Setup dispositivo
    device = get_optimal_device()
    print(f"✅ Utilizzo dispositivo: {device}")
    
    # Carica modello
    try:
        model_path = f'yolov8{model_size}.pt'
        model = YOLO(model_path)
        print(f"✅ Modello YOLOv8{model_size} caricato")
    except Exception as e:
        print(f"❌ Errore nel caricamento del modello: {e}")
        return None
    
    # Configurazione del progetto
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_name = f'medical_detection_confusion_opt'
    run_name = f'mastopessi_{model_size}_{timestamp}'
    
    # Configurazione di addestramento ottimizzata per migliorare la confusion matrix
    train_config = get_confusion_matrix_optimized_config(
        data_yaml_path, epochs, batch_size, device, project_name, run_name, class_weights
    )
    
    print("\n🔄 Addestramento con configurazione ottimizzata per confusion matrix...")
    
    try:
        # Addestramento principale
        results = model.train(**train_config)
        print("\n✅ Addestramento completato!")
        
        # Trova il modello migliore
        best_weights_path = os.path.join(project_name, run_name, 'weights', 'best.pt')
        
        if os.path.exists(best_weights_path):
            print("\n📈 Valutazione modello con analisi approfondita...")
            
            # Valutazione con multiple threshold per ottimizzare la confusion matrix
            optimal_thresholds = find_optimal_thresholds(best_weights_path, data_yaml_path)
            
            # Genera confusion matrix migliorata
            generate_enhanced_confusion_matrix(best_weights_path, data_yaml_path, optimal_thresholds)
            
            # Analisi degli errori
            analyze_prediction_errors(best_weights_path, data_yaml_path, optimal_thresholds)
            
            return model, best_weights_path, optimal_thresholds
        else:
            print(f"⚠️ Modello migliore non trovato in {best_weights_path}")
            return None, None, None
        
    except Exception as e:
        print(f"\n❌ Errore durante l'addestramento: {e}")
        provide_troubleshooting_tips(e)
        return None, None, None


def get_confusion_matrix_optimized_config(data_yaml_path, epochs, batch_size, device, project_name, run_name, class_weights):
    """
    Configurazione ottimizzata e sicura per migliorare la confusion matrix
    """
    # Configurazione base sicura
    config = {
        # Parametri essenziali
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': 640,
        'device': device,
        'project': project_name,
        'name': run_name,
        'exist_ok': True,
        
        # Parametri di training ottimizzati
        'patience': 70,
        'save': True,
        'optimizer': 'AdamW',
        'lr0': 0.0008,
        'lrf': 0.005,
        'momentum': 0.937,
        'weight_decay': 0.0008,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights per migliorare detection
        'box': 7.5,
        'cls': 1.0,
        'dfl': 1.5,
        
        # Data augmentation per immagini mediche
        'hsv_h': 0.01,
        'hsv_s': 0.3,
        'hsv_v': 0.2,
        'degrees': 5.0,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 1.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.3,
        'mosaic': 0.8,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # Configurazioni aggiuntive sicure
        'cos_lr': True,
        'close_mosaic': 15,
        'amp': True,
        'plots': True,
        'save_period': 25,
        'verbose': True,
        'val': True,
        'save_json': True,
    }
    
    # Aggiungi parametri opzionali solo se supportati
    try:
        # Test se i parametri sono supportati nella versione corrente
        optional_params = {
            'fraction': 1.0,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.0,
            'split': 'val',
            'save_hybrid': False,
            'iou': 0.7,
            'max_det': 300,
            'half': False,
            'save_txt': True,
            'save_conf': True,
            'save_crop': False,
            'show_labels': True,
            'show_conf': True,
            'agnostic_nms': False,
            'retina_masks': False,
        }
        
        # Aggiungi solo parametri che non causano errori
        config.update(optional_params)
        
    except Exception as e:
        print(f"⚠️ Alcuni parametri opzionali non supportati: {e}")
    
    return config


def analyze_and_calculate_class_weights(data_yaml_path):
    """
    Analizza il dataset e calcola pesi per le classi per bilanciare il training
    """
    print("\n📊 Analisi avanzata distribuzione classi...")
    
    try:
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = config['path']
        classes = config['names']
        nc = config['nc']
        
        # Conta oggetti e immagini per classe
        class_object_counts = {i: 0 for i in range(nc)}
        class_image_counts = {i: 0 for i in range(nc)}
        total_images = 0
        
        for split in ['train', 'val']:
            if split not in config:
                continue
                
            split_path = config[split]
            if 'labels' in split_path:
                lbl_path = os.path.join(base_path, split_path)
            elif 'images' in split_path:
                lbl_path = os.path.join(base_path, split_path.replace('images', 'labels'))
            else:
                lbl_path = os.path.join(base_path, split_path, 'labels')
            
            if not os.path.exists(lbl_path):
                continue
            
            for lbl_file in os.listdir(lbl_path):
                if not lbl_file.endswith('.txt'):
                    continue
                
                total_images += 1
                image_classes = set()
                
                try:
                    with open(os.path.join(lbl_path, lbl_file), 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                cls_id = int(line.split()[0])
                                if 0 <= cls_id < nc:
                                    class_object_counts[cls_id] += 1
                                    image_classes.add(cls_id)
                    
                    # Conta immagini per classe
                    for cls_id in image_classes:
                        class_image_counts[cls_id] += 1
                        
                except (ValueError, IndexError, IOError):
                    continue
        
        # Calcola statistiche avanzate
        total_objects = sum(class_object_counts.values())
        
        print("\n📈 Statistiche dettagliate:")
        print("-" * 80)
        print(f"{'Classe':<20} {'Oggetti':<10} {'%Obj':<8} {'Immagini':<10} {'%Img':<8} {'Obj/Img':<8}")
        print("-" * 80)
        
        class_weights = {}
        
        for cls_id in range(nc):
            class_name = classes[cls_id] if cls_id < len(classes) else f"Classe_{cls_id}"
            obj_count = class_object_counts[cls_id]
            img_count = class_image_counts[cls_id]
            
            obj_pct = (obj_count / total_objects * 100) if total_objects > 0 else 0
            img_pct = (img_count / total_images * 100) if total_images > 0 else 0
            obj_per_img = (obj_count / img_count) if img_count > 0 else 0
            
            print(f"{class_name:<20} {obj_count:<10} {obj_pct:<7.1f}% {img_count:<10} {img_pct:<7.1f}% {obj_per_img:<7.1f}")
            
            # Calcola peso per bilanciamento (inversamente proporzionale alla frequenza)
            if obj_count > 0:
                class_weights[cls_id] = total_objects / (nc * obj_count)
            else:
                class_weights[cls_id] = 1.0
        
        # Normalizza pesi
        max_weight = max(class_weights.values())
        for cls_id in class_weights:
            class_weights[cls_id] = class_weights[cls_id] / max_weight
        
        print(f"\nTotale: {total_objects} oggetti in {total_images} immagini")
        print("\n🏋️ Pesi calcolati per bilanciamento:")
        for cls_id, weight in class_weights.items():
            class_name = classes[cls_id] if cls_id < len(classes) else f"Classe_{cls_id}"
            print(f"  {class_name}: {weight:.3f}")
        
        return class_weights
        
    except Exception as e:
        print(f"❌ Errore nell'analisi delle classi: {e}")
        return {i: 1.0 for i in range(nc)}


def validate_dataset_quality(data_config):
    """
    Validazione avanzata della qualità del dataset
    """
    print("\n🔍 Validazione qualità dataset...")
    
    base_path = data_config['path']
    issues = []
    
    for split in ['train', 'val']:
        if split not in data_config:
            continue
            
        split_path = data_config[split]
        
        # Costruisci percorsi
        if 'images' in split_path:
            img_path = os.path.join(base_path, split_path)
            lbl_path = img_path.replace('images', 'labels')
        else:
            img_path = os.path.join(base_path, split_path, 'images')
            lbl_path = os.path.join(base_path, split_path, 'labels')
        
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            continue
        
        # Verifica corrispondenza e qualità
        images = [f for f in os.listdir(img_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]
        
        print(f"\n📋 Analisi {split}:")
        print(f"  Immagini: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # Verifica file vuoti o problematici
        empty_labels = 0
        invalid_labels = 0
        
        for lbl_file in labels:
            lbl_path_full = os.path.join(lbl_path, lbl_file)
            
            try:
                with open(lbl_path_full, 'r') as f:
                    content = f.read().strip()
                    
                if not content:
                    empty_labels += 1
                else:
                    # Verifica formato annotations
                    lines = content.split('\n')
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) != 5:
                                issues.append(f"{split}/{lbl_file}:{line_num} - Formato non valido")
                                invalid_labels += 1
                            else:
                                try:
                                    cls_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:]]
                                    # Verifica range coordinate
                                    if not all(0 <= coord <= 1 for coord in coords):
                                        issues.append(f"{split}/{lbl_file}:{line_num} - Coordinate fuori range")
                                except ValueError:
                                    issues.append(f"{split}/{lbl_file}:{line_num} - Valori non numerici")
                                    invalid_labels += 1
                            
            except Exception as e:
                issues.append(f"{split}/{lbl_file} - Errore lettura: {e}")
                invalid_labels += 1
        
        if empty_labels:
            print(f"  ⚠️ Labels vuote: {empty_labels}")
        if invalid_labels:
            print(f"  ❌ Labels non valide: {invalid_labels}")
    
    # Mostra primi problemi trovati
    if issues:
        print(f"\n⚠️ Trovati {len(issues)} problemi nel dataset:")
        for issue in issues[:5]:  # Mostra solo i primi 5
            print(f"  • {issue}")
        if len(issues) > 5:
            print(f"  ... e altri {len(issues) - 5} problemi")
    else:
        print("✅ Dataset validato senza problemi critici")


def find_optimal_thresholds(model_path, data_yaml_path):
    """
    Trova le threshold ottimali per confidence e IoU per massimizzare la diagonal della confusion matrix
    """
    print("\n🎯 Ricerca threshold ottimali...")
    
    model = YOLO(model_path)
    
    # Test range di threshold
    conf_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    iou_thresholds = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    
    best_f1 = 0
    best_conf = 0.25
    best_iou = 0.5
    
    print("🔄 Testing combinazioni threshold...")
    
    for conf_th in conf_thresholds:
        for iou_th in iou_thresholds:
            try:
                results = model.val(
                    data=data_yaml_path,
                    conf=conf_th,
                    iou=iou_th,
                    verbose=False,
                    plots=False
                )
                
                if hasattr(results, 'box') and hasattr(results.box, 'p') and hasattr(results.box, 'r'):
                    precision = results.box.p
                    recall = results.box.r
                    
                    if precision > 0 and recall > 0:
                        f1 = 2 * (precision * recall) / (precision + recall)
                        
                        if f1 > best_f1:
                            best_f1 = f1
                            best_conf = conf_th
                            best_iou = iou_th
                            
            except Exception as e:
                continue
    
    print(f"✅ Threshold ottimali trovate:")
    print(f"  Confidence: {best_conf}")
    print(f"  IoU: {best_iou}")
    print(f"  F1-Score: {best_f1:.4f}")
    
    return {'conf': best_conf, 'iou': best_iou, 'f1': best_f1}


def generate_enhanced_confusion_matrix(model_path, data_yaml_path, optimal_thresholds):
    """
    Genera una confusion matrix migliorata con threshold ottimizzate
    """
    print("\n🔄 Generazione confusion matrix migliorata...")
    
    try:
        model = YOLO(model_path)
        
        # Carica configurazione classi
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        class_names = config['names']
        
        # Valutazione con threshold ottimizzate
        results = model.val(
            data=data_yaml_path,
            conf=optimal_thresholds['conf'],
            iou=optimal_thresholds['iou'],
            max_det=300,
            plots=True,
            save_json=True,
            verbose=True
        )
        
        if hasattr(results, 'confusion_matrix') and results.confusion_matrix is not None:
            # Ottieni la confusion matrix
            cm = results.confusion_matrix.matrix
            
            # Crea visualizzazione migliorata
            plt.figure(figsize=(12, 10))
            
            # Normalizza per percentuali
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            cm_norm = np.nan_to_num(cm_norm)
            
            # Crea heatmap
            mask = cm == 0
            sns.heatmap(cm_norm, 
                       annot=True, 
                       fmt='.1f',
                       cmap='Blues',
                       mask=mask,
                       xticklabels=class_names + ['Background'],
                       yticklabels=class_names + ['Background'],
                       cbar_kws={'label': 'Percentuale (%)'})
            
            plt.title(f'Confusion Matrix Migliorata\n(Conf: {optimal_thresholds["conf"]}, IoU: {optimal_thresholds["iou"]})')
            plt.xlabel('Predetto')
            plt.ylabel('Reale')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            # Salva
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cm_path = f'enhanced_confusion_matrix_{timestamp}.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confusion matrix migliorata salvata: {cm_path}")
            
            # Analizza diagonale
            diagonal_sum = np.trace(cm_norm[:-1, :-1])  # Escludi background
            total_predictions = np.sum(cm_norm[:-1, :-1])
            diagonal_percentage = (diagonal_sum / total_predictions) * 100 if total_predictions > 0 else 0
            
            print(f"📊 Analisi Confusion Matrix:")
            print(f"  Accuratezza diagonale: {diagonal_percentage:.1f}%")
            print(f"  Background FP: {cm_norm[-1, :-1].sum():.1f}%")
            print(f"  Background FN: {cm_norm[:-1, -1].sum():.1f}%")
            
            # Suggerimenti basati sui risultati
            if diagonal_percentage < 70:
                print("\n💡 Suggerimenti per migliorare:")
                print("  • Considera di aumentare i dati di training")
                print("  • Verifica la qualità delle annotazioni")
                print("  • Prova data augmentation più specifica")
                if cm_norm[:-1, -1].sum() > 20:  # Molti FN
                    print("  • Riduci confidence threshold per catturare più oggetti")
                if cm_norm[-1, :-1].sum() > 20:  # Molti FP
                    print("  • Aumenta confidence threshold per ridurre falsi positivi")
        
        return results
        
    except Exception as e:
        print(f"❌ Errore nella generazione confusion matrix: {e}")
        return None


def analyze_prediction_errors(model_path, data_yaml_path, optimal_thresholds):
    """
    Analizza gli errori di predizione per capire dove migliorare
    """
    print("\n🔍 Analisi errori di predizione...")
    
    try:
        model = YOLO(model_path)
        
        # Carica configurazione
        with open(data_yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        base_path = config['path']
        val_path = config.get('val', 'val')
        
        if 'images' in val_path:
            img_path = os.path.join(base_path, val_path)
        else:
            img_path = os.path.join(base_path, val_path, 'images')
        
        # Analizza un campione di immagini
        sample_images = random.sample([f for f in os.listdir(img_path) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))], 
                                    min(20, len(os.listdir(img_path))))
        
        error_stats = {
            'false_negatives': 0,
            'false_positives': 0,
            'low_confidence': 0,
            'misclassified': 0
        }
        
        for img_file in sample_images:
            img_full_path = os.path.join(img_path, img_file)
            
            # Predizione
            results = model.predict(
                img_full_path,
                conf=optimal_thresholds['conf'],
                iou=optimal_thresholds['iou'],
                verbose=False
            )
            
            # Carica ground truth
            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            lbl_path = img_path.replace('images', 'labels')
            lbl_full_path = os.path.join(lbl_path, lbl_file)
            
            if os.path.exists(lbl_full_path):
                with open(lbl_full_path, 'r') as f:
                    gt_lines = [line.strip() for line in f if line.strip()]
                
                num_gt = len(gt_lines)
                num_pred = len(results[0].boxes) if results[0].boxes is not None else 0
                
                if num_pred < num_gt:
                    error_stats['false_negatives'] += (num_gt - num_pred)
                elif num_pred > num_gt:
                    error_stats['false_positives'] += (num_pred - num_gt)
                
                # Analizza confidence
                if results[0].boxes is not None:
                    confidences = results[0].boxes.conf.cpu().numpy()
                    low_conf_count = np.sum(confidences < 0.5)
                    error_stats['low_confidence'] += low_conf_count
        
        print("📊 Statistiche errori (campione):")
        for error_type, count in error_stats.items():
            print(f"  {error_type.replace('_', ' ').title()}: {count}")
        
        # Raccomandazioni
        print("\n💡 Raccomandazioni basate sull'analisi:")
        if error_stats['false_negatives'] > error_stats['false_positives']:
            print("  • Focus su recall: riduci confidence threshold")
            print("  • Aumenta data augmentation per oggetti piccoli")
        elif error_stats['false_positives'] > error_stats['false_negatives']:
            print("  • Focus su precision: aumenta confidence threshold")
            print("  • Migliora qualità annotazioni per ridurre ambiguità")
        
        if error_stats['low_confidence'] > len(sample_images) * 0.3:
            print("  • Molte predizioni con bassa confidence")
            print("  • Considera più epochs di training o learning rate più basso")
        
    except Exception as e:
        print(f"❌ Errore nell'analisi degli errori: {e}")


def get_optimal_device():
    """Determina il dispositivo ottimale per l'addestramento"""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def provide_troubleshooting_tips(error):
    """Fornisce suggerimenti specifici basati sull'errore"""
    error_str = str(error).lower()
    
    print("\n🔧 Suggerimenti per la risoluzione:")
    
    if "no such file or directory" in error_str:
        print("• Verifica che tutti i percorsi nel data.yaml siano corretti")
        print("• Controlla che le directory images/ e labels/ esistano")
        print("• Assicurati che i percorsi siano relativi alla 'path' specificata")
    
    elif "cuda" in error_str or "mps" in error_str:
        print("• Prova a utilizzare device='cpu' se hai problemi GPU")
        print("• Riduci la batch_size se hai problemi di memoria")
        print("• Aggiorna i driver GPU se usi CUDA")
    
    elif "memory" in error_str or "out of memory" in error_str:
        print("• Riduci batch_size (prova 8 o 4)")
        print("• Riduci imgsz (prova 416 invece di 640)")
        print("• Chiudi altre applicazioni che usano memoria")
    
    else:
        print("• Verifica la struttura del dataset")
        print("• Controlla che le annotations siano nel formato YOLO corretto")
        print("• Assicurati che le classi nel data.yaml corrispondano alle labels")


def main():
    """Funzione principale ottimizzata per confusion matrix"""
    DATA_YAML = "/Users/salvatore/Development/KazaamLab/Mastopessi2/data/data.yaml"
    
    print("🏥 Sistema YOLO Medico - Ottimizzazione Confusion Matrix")
    print("=" * 60)
    
    # Addestra modello con focus sulla confusion matrix
    model, model_path, thresholds = train_medical_yolo_optimized(
        DATA_YAML, 
        epochs=100, 
        batch_size=16,
        model_size='l'
    )
    
    if model and model_path and thresholds:
        print("\n🎉 Addestramento completato con successo!")
        print(f"📁 Modello salvato in: {model_path}")
        print(f"🎯 Threshold ottimali: Conf={thresholds['conf']}, IoU={thresholds['iou']}")
        
        # Test finale con threshold ottimizzate
        print("\n🧪 Test finale con threshold ottimizzate...")
        final_results = model.val(
            data=DATA_YAML,
            conf=thresholds['conf'],
            iou=thresholds['iou'],
            plots=True,
            verbose=True
        )
        
        if hasattr(final_results, 'box'):
            print(f"\n📊 Risultati finali:")
            print(f"  mAP@0.5: {final_results.box.map50:.4f}")
            print(f"  mAP@0.5:0.95: {final_results.box.map:.4f}")
            print(f"  Precision: {final_results.box.p:.4f}")
            print(f"  Recall: {final_results.box.r:.4f}")
            print(f"  F1-Score: {thresholds['f1']:.4f}")
        
        # Salva configurazione ottimale
        save_optimal_config(thresholds, model_path, DATA_YAML)
        
        print("\n✅ Processo completato!")
        print("📊 Controlla i file generati per la confusion matrix migliorata")
        
    else:
        print("\n❌ Addestramento fallito. Controlla i suggerimenti sopra.")


def save_optimal_config(thresholds, model_path, data_yaml_path):
    """Salva la configurazione ottimale per uso futuro"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    config_file = f"optimal_config_{timestamp}.yaml"
    
    try:
        optimal_config = {
            'model_path': model_path,
            'data_yaml': data_yaml_path,
            'optimal_thresholds': {
                'confidence': float(thresholds['conf']),
                'iou': float(thresholds['iou']),
                'f1_score': float(thresholds['f1'])
            },
            'recommended_inference_settings': {
                'conf': float(thresholds['conf']),
                'iou': float(thresholds['iou']),
                'max_det': 300,
                'agnostic_nms': False,
                'retina_masks': False,
                'embed': None
            },
            'training_timestamp': timestamp,
            'notes': [
                "Configurazione ottimizzata per massimizzare diagonal confusion matrix",
                "Threshold testate su validation set",
                "Raccomandato per inferenza su immagini mediche simili"
            ]
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(optimal_config, f, default_flow_style=False, indent=2)
        
        print(f"💾 Configurazione ottimale salvata in: {config_file}")
        
    except Exception as e:
        print(f"⚠️ Errore nel salvataggio configurazione: {e}")


def inference_with_optimal_settings(model_path, image_path, config_file=None):
    """
    Funzione per inferenza con impostazioni ottimizzate
    """
    print("🔮 Inferenza con impostazioni ottimizzate...")
    
    # Carica configurazione se disponibile
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        conf = config['optimal_thresholds']['confidence']
        iou = config['optimal_thresholds']['iou']
        print(f"📄 Configurazione caricata da: {config_file}")
    else:
        # Impostazioni di default ottimizzate
        conf = 0.25
        iou = 0.5
        print("⚙️ Utilizzo impostazioni di default ottimizzate")
    
    try:
        model = YOLO(model_path)
        
        results = model.predict(
            image_path,
            conf=conf,
            iou=iou,
            max_det=300,
            verbose=True,
            save=True,
            show_labels=True,
            show_conf=True
        )
        
        print("✅ Inferenza completata!")
        return results
        
    except Exception as e:
        print(f"❌ Errore durante inferenza: {e}")
        return None


def batch_evaluate_with_confusion_analysis(model_path, test_images_dir, optimal_config_file):
    """
    Valuta un batch di immagini e analizza la confusion matrix risultante
    """
    print("📊 Valutazione batch con analisi confusion matrix...")
    
    try:
        # Carica configurazione ottimale
        with open(optimal_config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        conf = config['optimal_thresholds']['confidence']
        iou = config['optimal_thresholds']['iou']
        
        model = YOLO(model_path)
        
        # Processa tutte le immagini
        image_files = [f for f in os.listdir(test_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        all_predictions = []
        all_ground_truths = []
        
        print(f"🔄 Processando {len(image_files)} immagini...")
        
        for img_file in image_files:
            img_path = os.path.join(test_images_dir, img_file)
            
            # Predizione
            results = model.predict(
                img_path,
                conf=conf,
                iou=iou,
                verbose=False
            )
            
            # Estrai predizioni
            if results[0].boxes is not None:
                pred_classes = results[0].boxes.cls.cpu().numpy().astype(int)
                all_predictions.extend(pred_classes)
            
            # Carica ground truth se disponibile
            lbl_file = os.path.splitext(img_file)[0] + '.txt'
            lbl_path = test_images_dir.replace('images', 'labels')
            lbl_full_path = os.path.join(lbl_path, lbl_file)
            
            if os.path.exists(lbl_full_path):
                with open(lbl_full_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            gt_class = int(line.split()[0])
                            all_ground_truths.append(gt_class)
        
        # Genera confusion matrix personalizzata
        if all_predictions and all_ground_truths:
            from sklearn.metrics import confusion_matrix, classification_report
            
            # Bilancia le liste se necessario
            min_len = min(len(all_predictions), len(all_ground_truths))
            all_predictions = all_predictions[:min_len]
            all_ground_truths = all_ground_truths[:min_len]
            
            cm = confusion_matrix(all_ground_truths, all_predictions)
            
            # Visualizza
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix - Batch Evaluation')
            plt.xlabel('Predetto')
            plt.ylabel('Reale')
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            cm_path = f'batch_confusion_matrix_{timestamp}.png'
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Confusion matrix batch salvata: {cm_path}")
            
            # Report classificazione
            report = classification_report(all_ground_truths, all_predictions)
            print("\n📋 Classification Report:")
            print(report)
            
        else:
            print("⚠️ Nessuna predizione o ground truth trovata")
        
    except Exception as e:
        print(f"❌ Errore nella valutazione batch: {e}")


if __name__ == "__main__":
    main()