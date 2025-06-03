# improved_training.py
from ultralytics import YOLO

def train_with_better_augmentation():
    """Training con augmentation migliorata per varietà di forme"""
    model = YOLO('yolov8l.pt')
    
    results = model.train(
        data='data/dataset.yaml',
        epochs=150,  # Più epoche
        imgsz=640,
        batch=16,
        device='mps',
        
        # Augmentation avanzata
        augment=True,
        degrees=30,  # Rotazione random
        translate=0.2,  # Traslazione
        scale=0.5,  # Scaling
        shear=20,  # Shear transformation
        perspective=0.0005,  # Prospettiva
        flipud=0.5,  # Flip verticale
        fliplr=0.5,  # Flip orizzontale
        mosaic=1.0,  # Mosaic augmentation
        mixup=0.1,  # Mixup augmentation
        copy_paste=0.1,  # Copy-paste augmentation
        
        # Altri parametri
        hsv_h=0.015,  # Hue augmentation
        hsv_s=0.7,  # Saturation augmentation  
        hsv_v=0.4,  # Value augmentation
        
        # Ottimizzazioni
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        close_mosaic=15,
        
        project='runs/train',
        name='breast_detection_improved',
        exist_ok=True,
        patience=30,
        save=True,
        plots=True
    )
    
    return model

# Esegui training migliorato
if __name__ == "__main__":
    model = train_with_better_augmentation()