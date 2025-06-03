# train_m4_corrected.py
from ultralytics import YOLO
import torch
import os

def train_model_m4_optimized():
    # Ottimizzazioni per Apple Silicon
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Verifica MPS
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("PyTorch non è compilato con supporto MPS")
            print("Installa PyTorch con: pip install torch torchvision torchaudio")
        device = 'cpu'
    else:
        device = 'mps'
    
    print(f"Usando dispositivo: {device}")
    
    # Modello large per sfruttare la potenza del M4
    model = YOLO('yolov8l.pt')  # l = large, M4 può gestirlo
    
    # Training ottimizzato per M4
    results = model.train(
        data='data/dataset.yaml',
        epochs=100,              
        imgsz=640,              
        batch=32,               
        patience=15,            
        device=device,
        workers=10,             
        project='runs/train',
        name='breast_detection_m4_optimized',
        exist_ok=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        augment=True,
        save=True,
        plots=True,
        cache='ram',            
        single_cls=True,        
        rect=True,              
        cos_lr=True,            
        close_mosaic=10,        
        resume=False,           
        amp=False,              
        fraction=1.0,           
        profile=False,          
        overlap_mask=True,      
        mask_ratio=4,           
        dropout=0.0,            
        val=True                
    )
    
    # Il modello viene salvato automaticamente durante il training
    # I pesi migliori sono in: runs/train/breast_detection_m4_optimized/weights/best.pt
    
    print("\nTraining completato con successo!")
    print(f"Modello salvato in: runs/train/breast_detection_m4_optimized/weights/best.pt")
    
    # Se vuoi esportare il modello in altri formati
    # model.export(format='onnx')  # Esporta in ONNX
    # model.export(format='coreml')  # Esporta per iOS
    
    return model

if __name__ == "__main__":
    # Info sistema
    print("Informazioni sistema:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS disponibile: {torch.backends.mps.is_available()}")
    if torch.backends.mps.is_available():
        print(f"MPS built: {torch.backends.mps.is_built()}")
    
    try:
        model = train_model_m4_optimized()
        
    except Exception as e:
        print(f"\nErrore durante il training: {e}")
        print("Prova a ridurre batch size o imgsz se hai problemi di memoria")