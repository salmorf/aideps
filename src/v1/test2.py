# use_model_correct.py
from ultralytics import YOLO
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# Carica il modello addestrato
model = YOLO('runs/train/breast_detection_improved/weights/best.pt')

# Metodo 1: Predizione e visualizzazione con matplotlib
def predict_and_show(image_path):
    # Esegui predizione
    results = model(image_path)
    
    # Ottieni l'immagine con le bounding box
    img_with_boxes = results[0].plot()
    
    # Converti BGR a RGB per matplotlib
    img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Mostra con matplotlib
    plt.figure(figsize=(12, 8))
    plt.imshow(img_rgb)
    plt.title(f'Predizione: {Path(image_path).name}')
    plt.axis('off')
    plt.show()
    
    # Salva il risultato
    cv2.imwrite('result.jpg', img_with_boxes)
    print("Risultato salvato come 'result.jpg'")
    
    return results

# Metodo 2: Visualizzazione con OpenCV
def predict_and_show_opencv(image_path):
    # Esegui predizione
    results = model(image_path)
    
    # Ottieni l'immagine con le bounding box
    img_with_boxes = results[0].plot()
    
    # Mostra con OpenCV
    cv2.imshow('Breast Detection', img_with_boxes)
    cv2.waitKey(0)  # Premi qualsiasi tasto per chiudere
    cv2.destroyAllWindows()
    
    return results

# Metodo 3: Ottenere informazioni dettagliate
def predict_with_details(image_path):
    # Esegui predizione
    results = model(image_path)
    
    # Estrai informazioni
    result = results[0]
    boxes = result.boxes
    
    print(f"\nRisultati per {Path(image_path).name}:")
    print(f"Numero di rilevamenti: {len(boxes)}")
    
    # Per ogni rilevamento
    for i, box in enumerate(boxes):
        conf = box.conf[0].item()  # Confidenza
        xyxy = box.xyxy[0].tolist()  # Coordinate [x1, y1, x2, y2]
        
        print(f"\nRilevamento {i+1}:")
        print(f"  Confidenza: {conf:.3f}")
        print(f"  Bounding box: {[int(x) for x in xyxy]}")
    
    # Visualizza l'immagine
    img_with_boxes = result.plot()
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
    plt.title('Breast Detection Results')
    plt.axis('off')
    plt.show()
    
    return results

# Script principale
if __name__ == "__main__":
    # Sostituisci con il percorso della tua immagine
    image_path = '/Users/salvatore/Development/KazaamLab/Mastopessi2/dataset/paziente_014.jpg'
    
    # Usa uno dei metodi
    results = predict_with_details(image_path)
    
    # Metodi alternativi per salvare/visualizzare
    # 1. Salva direttamente
    # results[0].save('output.jpg')
    
    # 2. Ottieni array numpy dell'immagine annotata
    # annotated_img = results[0].plot()
    
    # 3. Ottieni le bounding box
    boxes = results[0].boxes
    for box in boxes:
        print(f"Confidenza: {box.conf[0]}, Box: {box.xyxy[0]}")