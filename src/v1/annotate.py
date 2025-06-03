import cv2
import os
from pathlib import Path

class BoundingBoxAnnotator:
    def __init__(self, image_path, output_dir):
        self.image_path = image_path
        self.output_dir = output_dir
        self.image = cv2.imread(image_path)
        self.clone = self.image.copy()
        self.bbox = []
        self.drawing = False
        
    def draw_rectangle(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.bbox = [(x, y)]
            self.drawing = True
            
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.clone = self.image.copy()
            cv2.rectangle(self.clone, self.bbox[0], (x, y), (0, 255, 0), 2)
            
        elif event == cv2.EVENT_LBUTTONUP:
            self.bbox.append((x, y))
            self.drawing = False
            cv2.rectangle(self.clone, self.bbox[0], self.bbox[1], (0, 255, 0), 2)
    
    def annotate(self):
        cv2.namedWindow("Annotate")
        cv2.setMouseCallback("Annotate", self.draw_rectangle)
        
        while True:
            cv2.imshow("Annotate", self.clone)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord("r"):  # Reset
                self.clone = self.image.copy()
                self.bbox = []
                
            elif key == ord("s"):  # Save
                if len(self.bbox) == 2:
                    self.save_annotation()
                    break
                    
            elif key == ord("q"):  # Quit
                break
        
        cv2.destroyAllWindows()
    
    def save_annotation(self):
        # Converti in formato YOLO
        img_height, img_width = self.image.shape[:2]
        x1, y1 = self.bbox[0]
        x2, y2 = self.bbox[1]
        
        # Calcola centro, larghezza e altezza normalizzati
        x_center = ((x1 + x2) / 2) / img_width
        y_center = ((y1 + y2) / 2) / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        # Salva annotazione
        label_path = Path(self.output_dir) / f"{Path(self.image_path).stem}.txt"
        with open(label_path, 'w') as f:
            f.write(f"0 {x_center} {y_center} {width} {height}\n")
        
        print(f"Annotation saved to {label_path}")

# Esempio di utilizzo
if __name__ == "__main__":
    image_dir = "data/images/train"
    label_dir = "data/labels/train"
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith(('.jpg', '.png')):
            img_path = os.path.join(image_dir, img_file)
            annotator = BoundingBoxAnnotator(img_path, label_dir)
            annotator.annotate()