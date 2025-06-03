import cv2
import numpy as np
from pathlib import Path
import threading
import queue
import time

class DualBreastAnnotator:
    def __init__(self, image_dir='data/raw_images', output_dir='data/raw_labels'):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        
        # Crea directory se non esistono
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Definisci le classi
        self.class_names = ['seno_destro', 'seno_sinistro', 'areola_destra', 'areola_sinistra', 
                           'giugulare_dx', 'giugulare_sx', 'capezzolo_dx', 'capezzolo_sx']
        self.current_class = 0  # Classe corrente selezionata
        
        # Colori per ogni classe
        self.class_colors = [
            (0, 255, 0),    # Verde per seno_destro
            (255, 0, 0),    # Blu per seno_sinistro
            (0, 255, 255),  # Giallo per areola_destra
            (255, 0, 255),  # Magenta per areola_sinistra
            (0, 128, 255),  # Arancione per giugulare_dx
            (0, 69, 255),   # Arancione scuro per giugulare_sx
            (128, 0, 128),  # Viola per capezzolo_dx
            (0, 0, 255),    # Rosso per capezzolo_sx
        ]
        
        # Trova tutte le immagini
        self.images = sorted(list(self.image_dir.glob('*.jpg')) + 
                           list(self.image_dir.glob('*.png')) +
                           list(self.image_dir.glob('*.jpeg')))
        
        if not self.images:
            print(f"Nessuna immagine trovata in {self.image_dir}")
            exit(1)
            
        self.current_idx = 0
        self.current_image = None
        self.display_image = None
        
        # Stato del disegno
        self.drawing = False
        self.start_point = None
        self.current_point = None
        self.rectangles = []  # Lista per TUTTI i rettangoli [(class_id, start_point, end_point), ...]
        
        # Threading
        self.mouse_queue = queue.Queue()
        self.render_queue = queue.Queue(maxsize=2)
        self.running = True
        self.render_thread = None
        self.mouse_thread = None
        
        # Lock per proteggere l'accesso ai dati condivisi
        self.data_lock = threading.Lock()
        
        print(f"Trovate {len(self.images)} immagini in {self.image_dir}")
    
    def mouse_callback(self, event, x, y, flags, param):
        # Metti gli eventi del mouse nella coda per processarli in un thread separato
        self.mouse_queue.put((event, x, y, flags, param))
    
    def mouse_processor(self):
        """Thread separato per processare eventi del mouse"""
        while self.running:
            try:
                # Attendi un evento del mouse con timeout
                event, x, y, flags, param = self.mouse_queue.get(timeout=0.1)
                
                with self.data_lock:
                    if event == cv2.EVENT_LBUTTONDOWN:
                        self.drawing = True
                        self.start_point = (x, y)
                        self.current_point = (x, y)
                        
                    elif event == cv2.EVENT_MOUSEMOVE:
                        if self.drawing:
                            self.current_point = (x, y)
                            # Richiedi un aggiornamento del display
                            self.request_render()
                            
                    elif event == cv2.EVENT_LBUTTONUP:
                        if self.drawing:
                            self.drawing = False
                            self.current_point = (x, y)
                            
                            # Aggiungi il nuovo rettangolo se ha dimensioni ragionevoli
                            if self.start_point and self.current_point:
                                x1, y1 = self.start_point
                                x2, y2 = self.current_point
                                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                                    # AGGIUNGI il rettangolo con la classe corrente
                                    self.rectangles.append((self.current_class, self.start_point, self.current_point))
                                    
                                    # Limita a massimo 8 rettangoli (uno per ogni tipo)
                                    if len(self.rectangles) > 8:
                                        self.rectangles = self.rectangles[-8:]
                            
                            self.request_render()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Errore nel mouse processor: {e}")
    
    def render_processor(self):
        """Thread separato per il rendering"""
        last_render_time = time.time()
        min_render_interval = 1.0 / 60  # 60 FPS massimo
        
        while self.running:
            try:
                # Attendi una richiesta di rendering con timeout
                _ = self.render_queue.get(timeout=0.1)
                
                # Limita il framerate
                current_time = time.time()
                elapsed = current_time - last_render_time
                if elapsed < min_render_interval:
                    time.sleep(min_render_interval - elapsed)
                
                # Esegui il rendering
                self.render_display()
                last_render_time = time.time()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Errore nel render processor: {e}")
    
    def request_render(self):
        """Richiedi un aggiornamento del display"""
        try:
            self.render_queue.put_nowait(True)
        except queue.Full:
            pass  # Ignora se la coda è piena
    
    def render_display(self):
        """Aggiorna il display (chiamato dal thread di rendering)"""
        if self.current_image is None:
            return
        
        with self.data_lock:
            # Copia i dati necessari per evitare problemi di concorrenza
            rectangles_copy = self.rectangles.copy()
            drawing = self.drawing
            start_point = self.start_point
            current_point = self.current_point
            current_class = self.current_class
        
        # Crea nuova immagine per il display
        display_image = self.current_image.copy()
        
        # Disegna tutti i rettangoli salvati con colori diversi per classe
        for rect in rectangles_copy:
            class_id, pt1, pt2 = rect
            color = self.class_colors[class_id]
            cv2.rectangle(display_image, pt1, pt2, color, 2)
            
            # Etichetta per identificare la classe
            label = self.class_names[class_id]
            cv2.putText(display_image, label, (pt1[0], pt1[1]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Disegna il rettangolo che si sta disegnando
        if drawing and start_point and current_point:
            color = self.class_colors[current_class]
            cv2.rectangle(display_image, start_point, current_point, color, 2)
            # Mostra quale classe si sta disegnando
            cv2.putText(display_image, f"Disegnando: {self.class_names[current_class]}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Aggiungi indicatore della classe corrente
        status_text = f"Classe corrente: {self.class_names[current_class]} (premi 1-4 per cambiare)"
        cv2.putText(display_image, status_text, (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.class_colors[current_class], 2)
        
        # Aggiorna l'immagine display in modo thread-safe
        with self.data_lock:
            self.display_image = display_image
    
    def save_annotation(self):
        if self.rectangles:
            # Ottieni dimensioni immagine originale
            height, width = self.current_image.shape[:2]
            
            # Salva annotazione
            img_name = self.images[self.current_idx].stem
            label_path = self.output_dir / f"{img_name}.txt"
            
            with open(label_path, 'w') as f:
                # Salva TUTTI i rettangoli con la loro classe
                for rect in self.rectangles:
                    class_id, start_point, end_point = rect
                    
                    # Estrai coordinate
                    x1, y1 = start_point
                    x2, y2 = end_point
                    
                    # Assicurati che le coordinate siano ordinate
                    x1, x2 = min(x1, x2), max(x1, x2)
                    y1, y2 = min(y1, y2), max(y1, y2)
                    
                    # Limita le coordinate all'interno dell'immagine
                    x1 = max(0, min(x1, width))
                    x2 = max(0, min(x2, width))
                    y1 = max(0, min(y1, height))
                    y2 = max(0, min(y2, height))
                    
                    # Converti in formato YOLO (normalizzato)
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    
                    # Scrivi una riga per ogni rettangolo con la classe corretta
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
            
            print(f"Salvata annotazione per {self.images[self.current_idx].name} - {len(self.rectangles)} rettangoli")
            return True
        return False
    
    def load_image(self):
        if 0 <= self.current_idx < len(self.images):
            img_path = self.images[self.current_idx]
            self.current_image = cv2.imread(str(img_path))
            
            # Reset rettangoli
            with self.data_lock:
                self.rectangles = []
            
            # Controlla se esiste già un'annotazione
            label_path = self.output_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                self.load_existing_annotation(label_path)
            
            self.request_render()
            return True
        return False
    
    def load_existing_annotation(self, label_path):
        """Carica e visualizza annotazioni esistenti"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        rectangles = []
        for line in lines:
            line = line.strip()
            if line:
                parts = line.split()
                if len(parts) == 5:
                    class_id, x_center, y_center, w, h = map(float, parts)
                    class_id = int(class_id)
                    height, width = self.current_image.shape[:2]
                    
                    # Converti da YOLO a coordinate pixel
                    x_center *= width
                    y_center *= height
                    w *= width
                    h *= height
                    
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    rectangles.append((class_id, (x1, y1), (x2, y2)))
        
        with self.data_lock:
            self.rectangles = rectangles
    
    def run(self):
        cv2.namedWindow("Dual Breast Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Dual Breast Annotator", self.mouse_callback)
        
        # Avvia i thread
        self.mouse_thread = threading.Thread(target=self.mouse_processor, daemon=True)
        self.render_thread = threading.Thread(target=self.render_processor, daemon=True)
        self.mouse_thread.start()
        self.render_thread.start()
        
        self.load_image()
        
        print("\nCOMANDI:")
        print("- Click e trascina: Disegna rettangolo per la classe corrente")
        print("- 1-8: Seleziona classe da annotare:")
        print("  1: seno_destro")
        print("  2: seno_sinistro")
        print("  3: areola_destra")
        print("  4: areola_sinistra")
        print("  5: giugulare_dx")
        print("  6: giugulare_sx")
        print("  7: capezzolo_dx")
        print("  8: capezzolo_sx")
        print("- S: Salva annotazione e passa alla prossima")
        print("- N: Prossima immagine (senza salvare)")
        print("- P: Immagine precedente")
        print("- C: Cancella tutti i rettangoli")
        print("- U: Cancella ultimo rettangolo")
        print("- Q: Esci")
        
        try:
            while True:
                with self.data_lock:
                    current_display = self.display_image
                
                if current_display is not None:
                    # Aggiungi informazioni
                    info_img = current_display.copy()
                    h, w = info_img.shape[:2]
                    
                    # Testo informativo
                    text = f"Immagine {self.current_idx + 1}/{len(self.images)} - {self.images[self.current_idx].name}"
                    cv2.putText(info_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Stato
                    with self.data_lock:
                        num_rectangles = len(self.rectangles)
                        # Conta quante annotazioni per ogni classe
                        class_counts = [0, 0, 0, 0, 0, 0, 0, 0]
                        for rect in self.rectangles:
                            class_counts[rect[0]] += 1
                    
                    status = f"Rettangoli: {num_rectangles}/8"
                    cv2.putText(info_img, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
                    
                    # Mostra conteggio per classe (diviso in due colonne)
                    for i, name in enumerate(self.class_names[:4]):
                        count_text = f"{name}: {class_counts[i]}"
                        color = self.class_colors[i]
                        cv2.putText(info_img, count_text, (10, 150 + i * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # Seconda colonna per le altre 4 classi
                    for i, name in enumerate(self.class_names[4:]):
                        count_text = f"{name}: {class_counts[i+4]}"
                        color = self.class_colors[i+4]
                        cv2.putText(info_img, count_text, (300, 150 + i * 25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
                    
                    # Comandi in basso
                    commands = "S=Salva | N=Prossima | P=Precedente | C=Cancella tutto | U=Cancella ultimo | Q=Esci | 1-8=Cambia classe"
                    cv2.putText(info_img, commands, (10, h - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Mostra
                    cv2.imshow("Dual Breast Annotator", info_img)
                
                key = cv2.waitKey(30) & 0xFF  # Aumentato il timeout per ridurre il carico sulla CPU
                
                if key == ord('q'):  # Esci
                    break
                elif key == ord('s'):  # Salva
                    if self.save_annotation():
                        # Passa automaticamente alla prossima
                        if self.current_idx < len(self.images) - 1:
                            self.current_idx += 1
                            self.load_image()
                        else:
                            print("Ultima immagine raggiunta!")
                elif key == ord('n'):  # Prossima senza salvare
                    if self.current_idx < len(self.images) - 1:
                        self.current_idx += 1
                        self.load_image()
                elif key == ord('p'):  # Precedente
                    if self.current_idx > 0:
                        self.current_idx -= 1
                        self.load_image()
                elif key == ord('c'):  # Cancella tutto
                    with self.data_lock:
                        self.rectangles = []
                    self.request_render()
                elif key == ord('u'):  # Cancella ultimo
                    with self.data_lock:
                        if self.rectangles:
                            self.rectangles.pop()
                    self.request_render()
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8')]:  # Cambia classe
                    new_class = key - ord('1')
                    if 0 <= new_class < len(self.class_names):
                        with self.data_lock:
                            self.current_class = new_class
                        self.request_render()
                        print(f"Classe corrente: {self.class_names[new_class]}")
        
        finally:
            # Ferma i thread
            self.running = False
            if self.mouse_thread:
                self.mouse_thread.join(timeout=1)
            if self.render_thread:
                self.render_thread.join(timeout=1)
            
            cv2.destroyAllWindows()
            print(f"\nAnnotazioni salvate in: {self.output_dir}")


if __name__ == "__main__":
    annotator = DualBreastAnnotator()
    annotator.run()