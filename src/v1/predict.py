from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class BreastDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        
    def predict(self, image_path, conf_threshold=0.5):
        """
        Esegue la predizione su un'immagine
        """
        # Predizione
        results = self.model(image_path, conf=conf_threshold)
        
        # Estrai risultati
        image = cv2.imread(image_path)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Coordinate del bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence)
                })
                
                # Disegna sul'immagine
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(image, f'Breast: {confidence:.2f}', (int(x1), int(y1)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, detections
    
    def predict_batch(self, image_folder, output_folder):
        """
        Predizione su un batch di immagini
        """
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        for img_path in Path(image_folder).glob('*.jpg'):
            annotated_img, detections = self.predict(str(img_path))
            output_path = Path(output_folder) / f"predicted_{img_path.name}"
            cv2.imwrite(str(output_path), annotated_img)
            print(f"Processed: {img_path.name}")

# Esempio di utilizzo
if __name__ == "__main__":
    detector = BreastDetector('../models/breast_detector.pt')
    
    # Predizione singola
    image_path = 'test_image.jpg'
    annotated_image, detections = detector.predict(image_path)
    cv2.imwrite('result.jpg', annotated_image)
    
    # Predizione batch
    detector.predict_batch('../data/test', '../results/predictions')