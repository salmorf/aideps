# detection_gui.py
import tkinter as tk
from tkinter import filedialog, Scale, BooleanVar, DoubleVar, StringVar
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
from v2.check_dataset import ImprovedMedicalDetector


class DetectionGUI:
    """Interfaccia grafica per il rilevatore anatomico"""
    
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Rilevatore Anatomico - Mastopessi")
        self.root.geometry("1280x800")
        
        # Carica il modello
        self.detector = ImprovedMedicalDetector(model_path)
        
        # Variabili
        self.current_image_path = None
        self.display_image = None
        self.original_image = None
        
        # Crea interfaccia
        self._create_ui()
    
    def _create_ui(self):
        """Crea l'interfaccia utente"""
        # Frame principale
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Dividi in due colonne: controlli a sinistra, immagine a destra
        controls_frame = ttk.LabelFrame(main_frame, text="Controlli")
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        image_frame = ttk.LabelFrame(main_frame, text="Visualizzazione")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas per l'immagine
        self.canvas = tk.Canvas(image_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame per i controlli
        # 1. Controlli file
        file_frame = ttk.LabelFrame(controls_frame, text="File")
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(file_frame, text="Apri Immagine", command=self._open_image).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(file_frame, text="Salva Risultato", command=self._save_image).pack(fill=tk.X, padx=5, pady=5)
        
        # 2. Controlli rilevamento
        detection_frame = ttk.LabelFrame(controls_frame, text="Rilevamento")
        detection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Soglia confidenza
        ttk.Label(detection_frame, text="Soglia Confidenza:").pack(anchor=tk.W, padx=5, pady=2)
        self.conf_var = DoubleVar(value=0.5)
        conf_scale = Scale(detection_frame, from_=0.1, to=0.9, resolution=0.05,
                          orient=tk.HORIZONTAL, variable=self.conf_var)
        conf_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Checkbox per filtri
        self.filter_var = BooleanVar(value=True)
        ttk.Checkbutton(detection_frame, text="Abilita Filtri Anti Falsi Positivi", 
                       variable=self.filter_var).pack(anchor=tk.W, padx=5, pady=5)
        
        # 3. Parametri avanzati
        advanced_frame = ttk.LabelFrame(controls_frame, text="Parametri Avanzati")
        advanced_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Min area ratio
        ttk.Label(advanced_frame, text="Area Min (% immagine):").pack(anchor=tk.W, padx=5, pady=2)
        self.min_area_var = DoubleVar(value=self.detector.min_area_ratio * 100)
        min_area_scale = Scale(advanced_frame, from_=0.1, to=10.0, resolution=0.1,
                              orient=tk.HORIZONTAL, variable=self.min_area_var)
        min_area_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Max area ratio
        ttk.Label(advanced_frame, text="Area Max (% immagine):").pack(anchor=tk.W, padx=5, pady=2)
        self.max_area_var = DoubleVar(value=self.detector.max_area_ratio * 100)
        max_area_scale = Scale(advanced_frame, from_=5.0, to=50.0, resolution=1.0,
                              orient=tk.HORIZONTAL, variable=self.max_area_var)
        max_area_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # Symmetry threshold
        ttk.Label(advanced_frame, text="Threshold Simmetria:").pack(anchor=tk.W, padx=5, pady=2)
        self.symm_var = DoubleVar(value=self.detector.symmetry_threshold)
        symm_scale = Scale(advanced_frame, from_=0.1, to=1.0, resolution=0.05,
                          orient=tk.HORIZONTAL, variable=self.symm_var)
        symm_scale.pack(fill=tk.X, padx=5, pady=2)
        
        # 4. Bottoni azione
        action_frame = ttk.Frame(controls_frame)
        action_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(action_frame, text="Rileva", command=self._detect).pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(action_frame, text="Reset Parametri", command=self._reset_params).pack(fill=tk.X, padx=5, pady=5)
        
        # 5. Log output
        log_frame = ttk.LabelFrame(controls_frame, text="Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Visualizzazione
        view_frame = ttk.Frame(controls_frame)
        view_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.view_var = StringVar(value="comparison")
        ttk.Radiobutton(view_frame, text="Confronto", variable=self.view_var, 
                       value="comparison", command=self._update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Solo Filtrato", variable=self.view_var, 
                       value="filtered", command=self._update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_frame, text="Solo Originale", variable=self.view_var, 
                       value="original", command=self._update_view).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = StringVar(value="Pronto")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Log iniziale
        self._log("Sistema inizializzato. Apri un'immagine per iniziare.")
    
    def _log(self, message):
        """Aggiunge un messaggio al log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
    
    def _open_image(self):
        """Apre un'immagine"""
        file_path = filedialog.askopenfilename(
            title="Seleziona un'immagine",
            filetypes=[("Immagini", "*.jpg *.jpeg *.png"), ("Tutti i file", "*.*")]
        )
        
        if not file_path:
            return
        
        try:
            self.current_image_path = file_path
            self.original_image = cv2.imread(file_path)
            
            if self.original_image is None:
                self._log(f"Errore: Impossibile leggere l'immagine {file_path}")
                return
            
            self._log(f"Immagine caricata: {os.path.basename(file_path)}")
            self.status_var.set(f"Immagine: {os.path.basename(file_path)}")
            
            # Mostra immagine
            self._show_image(self.original_image)
            
            # Rileva automaticamente
            self._detect()
            
        except Exception as e:
            self._log(f"Errore nel caricamento dell'immagine: {e}")
    
    def _save_image(self):
        """Salva l'immagine risultato"""
        if self.display_image is None:
            self._log("Nessuna immagine da salvare")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Salva risultato",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Tutti i file", "*.*")],
            initialfile=f"result_{os.path.basename(self.current_image_path)}" if self.current_image_path else "result.jpg"
        )
        
        if not save_path:
            return
        
        try:
            cv2.imwrite(save_path, cv2.cvtColor(self.display_image, cv2.COLOR_RGB2BGR))
            self._log(f"Immagine salvata in: {save_path}")
        except Exception as e:
            self._log(f"Errore nel salvataggio dell'immagine: {e}")
    
    def _detect(self):
        """Esegue la detection sull'immagine corrente"""
        if self.original_image is None or self.current_image_path is None:
            self._log("Apri prima un'immagine")
            return
        
        try:
            # Aggiorna parametri
            self.detector.adjust_parameters(
                confidence=self.conf_var.get(),
                enable_filters=self.filter_var.get(),
                min_area_ratio=self.min_area_var.get() / 100,
                max_area_ratio=self.max_area_var.get() / 100,
                symmetry_threshold=self.symm_var.get()
            )
            
            # Crea una copia per non modificare l'originale
            self._log("Esecuzione rilevamento...")
            
            # Ridireziona output log
            self._redirect_detector_output()
            
            # Esegui detection
            filtered_img, filtered_detections = self.detector.detect(
                self.current_image_path, visualize=False
            )
            
            # Genera versione combinata
            raw_img = self.original_image.copy()
            self.detector._draw_detections(raw_img, self.detector._parse_detections(
                self.detector.model(self.original_image, conf=self.conf_var.get()), 
                self.original_image.shape
            ), "Senza Filtro")
            
            # Vista comparison
            h, w = self.original_image.shape[:2]
            combined = np.zeros((h, w*2, 3), dtype=np.uint8)
            combined[:, :w] = raw_img
            combined[:, w:] = filtered_img
            
            # Salva le varie versioni
            self.result_raw = raw_img
            self.result_filtered = filtered_img
            self.result_combined = combined
            
            # Aggiorna vista in base alla selezione
            self._update_view()
            
            self._log(f"Rilevamento completato con {len(filtered_detections)} oggetti")
            self.status_var.set(f"Rilevati {len(filtered_detections)} oggetti")
            
        except Exception as e:
            self._log(f"Errore nel rilevamento: {e}")
    
    def _reset_params(self):
        """Resetta i parametri ai valori di default"""
        self.conf_var.set(0.5)
        self.filter_var.set(True)
        self.min_area_var.set(0.5)
        self.max_area_var.set(30.0)
        self.symm_var.set(0.3)
        
        self._log("Parametri resettati ai valori di default")
    
    def _show_image(self, img):
        """Mostra un'immagine nel canvas"""
        if img is None:
            return
        
        # Converti da BGR a RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img
            
        self.display_image = img_rgb
            
        # Ridimensiona mantenendo aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas non ancora renderizzato, usa dimensioni predefinite
            canvas_width = 800
            canvas_height = 600
        
        h, w = img_rgb.shape[:2]
        ratio = min(canvas_width/w, canvas_height/h)
        new_size = (int(w*ratio), int(h*ratio))
        
        # Ridimensiona
        img_resized = cv2.resize(img_rgb, new_size, interpolation=cv2.INTER_AREA)
        
        # Converti per tkinter
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_resized))
        
        # Mostra nel canvas
        self.canvas.config(width=new_size[0], height=new_size[1])
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
        self.canvas.image = img_tk  # Salva riferimento
    
    def _update_view(self):
        """Aggiorna la visualizzazione in base alla selezione"""
        view_mode = self.view_var.get()
        
        if hasattr(self, 'result_raw') and hasattr(self, 'result_filtered') and hasattr(self, 'result_combined'):
            if view_mode == "comparison":
                self._show_image(self.result_combined)
            elif view_mode == "filtered":
                self._show_image(self.result_filtered)
            elif view_mode == "original":
                self._show_image(self.result_raw)
    
    def _redirect_detector_output(self):
        """Ridireziona l'output del detector al log"""
        import sys
        from io import StringIO
        
        class LogRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
                
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
                self.text_widget.update_idletasks()
                
            def flush(self):
                pass
        
        old_stdout = sys.stdout
        sys.stdout = LogRedirector(self.log_text)
        
        self.root.after(100, lambda: setattr(sys, 'stdout', old_stdout))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Interfaccia per rilevamento anatomico')
    parser.add_argument('--model', type=str, required=True, help='Percorso al modello')
    
    args = parser.parse_args()
    
    root = tk.Tk()
    app = DetectionGUI(root, args.model)
    root.mainloop()