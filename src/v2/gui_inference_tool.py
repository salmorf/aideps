# gui_inference_tool.py
import gradio as gr
import os
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime


class MedicalYOLOInference:
    def __init__(self):
        self.model = None
        self.class_colors = {
            'seno_destro': (255, 0, 0),      # Blu
            'seno_sinistro': (0, 255, 0),    # Verde
            'areola_destra': (0, 0, 255),    # Rosso
            'areola_sinistra': (255, 255, 0), # Ciano
            'capezzolo_dx': (255, 0, 255),   # Magenta
            'capezzolo_sx': (128, 0, 255),   # Viola
            'giugulare_dx': (0, 128, 255),   # Azzurro
            'giugulare_sx': (255, 128, 0)    # Arancione
        }
        self.corrections_applied = 0
        self.anatomical_correction_enabled = True
        
    def load_model(self, model_path):
        """Carica il modello YOLO"""
        try:
            self.model = YOLO(model_path)
            
            print("Classi del modello:")
            for class_id, class_name in self.model.names.items():
                print(f"  ID {class_id}: {class_name}")
                
            return f"✅ Modello caricato: {os.path.basename(model_path)}"
        except Exception as e:
            return f"❌ Errore nel caricamento del modello: {str(e)}"
    
    def analyze_image(self, image, confidence_threshold, model_path=None):
        """Analizza un'immagine e restituisce i risultati"""
        if self.model is None or model_path is not None:
            if model_path is None:
                return (image, "❌ Nessun modello caricato. Carica prima un modello.")
            self.load_model(model_path)
            
        if self.model is None:
            return (image, "❌ Modello non caricato correttamente.")
        
        if isinstance(image, np.ndarray):
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            return (image, "❌ Formato immagine non valido.")
        
        temp_path = "temp_image.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        
        try:
            results = self.model(temp_path, conf=confidence_threshold, iou=0.7, agnostic_nms=True, max_det=10)
        except Exception as e:
            os.remove(temp_path)
            return (image, f"❌ Errore durante l'inferenza: {str(e)}")
        
        detection_info = []
        result_image = image_rgb.copy()
        
        self.corrections_applied = 0
        
        print("Classi disponibili nel modello:", self.model.names)
        
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                if cls in self.model.names:
                    class_name = self.model.names[cls]
                else:
                    class_name = f"classe_{cls}"
                
                print(f"Class ID: {cls}, Class Name: {class_name}, Confidence: {conf:.2f}, Coords: ({x1},{y1})-({x2},{y2})")
                
                all_detections.append({
                    'class': class_name,
                    'class_id': cls,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2)
                })
        
        all_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        def calculate_iou(box1, box2):
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0  
            
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            
            union_area = box1_area + box2_area - intersection_area
            
            iou = intersection_area / union_area if union_area > 0 else 0
            return iou
            
        def apply_anatomical_correction(detections):
            if not self.anatomical_correction_enabled:
                print("Correzione anatomica disabilitata dall'utente")
                return detections
                
            corrected_detections = []
            
            img_height, img_width = image_rgb.shape[:2]
            img_center_x = img_width / 2
            
            print(f"Centro immagine: x={img_center_x}")
            
            correction_map = {
                'areola_destra': 'areola_sinistra',
                'areola_sinistra': 'areola_destra',
                'seno_destro': 'seno_sinistro',
                'seno_sinistro': 'seno_destro',
                'capezzolo_dx': 'capezzolo_sx',
                'capezzolo_sx': 'capezzolo_dx',
                'giugulare_dx': 'giugulare_sx',
                'giugulare_sx': 'giugulare_dx'
            }
            
            class_id_map = {}
            for id, name in self.model.names.items():
                for original, corrected in correction_map.items():
                    if name == original:
                        for corr_id, corr_name in self.model.names.items():
                            if corr_name == corrected:
                                class_id_map[id] = corr_id
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                
                box_center_x = (x1 + x2) / 2
                
                class_name = det['class']
                class_id = det['class_id']
                
                on_left_side = box_center_x < img_center_x
                
                should_correct = False
                
                if ('destra' in class_name or '_dx' in class_name) and not on_left_side:
                    should_correct = True
                    print(f"Correzione: {class_name} rilevato nel lato destro dell'immagine")
                elif ('sinistra' in class_name or '_sx' in class_name) and on_left_side:
                    should_correct = True
                    print(f"Correzione: {class_name} rilevato nel lato sinistro dell'immagine")
                
                # Se è necessaria una correzione, cambia la classe
                if should_correct and class_name in correction_map:
                    corrected_name = correction_map[class_name]
                    corrected_id = class_id_map.get(class_id, class_id)
                    
                    print(f"Correzione applicata: {class_name} -> {corrected_name}")
                    self.corrections_applied += 1
                    
                    det_copy = det.copy()
                    det_copy['class'] = corrected_name
                    det_copy['class_id'] = corrected_id
                    corrected_detections.append(det_copy)
                else:
                    corrected_detections.append(det)
            
            return corrected_detections
        
        all_detections = apply_anatomical_correction(all_detections)
        
        # Filtra i rilevamenti per evitare sovrapposizioni eccessive
        filtered_detections = []
        for det in all_detections:
            bbox = det['bbox']
            is_duplicate = False
            
            for existing_det in filtered_detections:
                existing_bbox = existing_det['bbox']
                iou = calculate_iou(bbox, existing_bbox)
                
                if iou > 0.5:  # Soglia IoU per considerare due rilevamenti come duplicati
                    # È un duplicato, lo saltiamo
                    is_duplicate = True
                    print(f"Filtrato rilevamento duplicato: {det['class']} (sovrapposto con {existing_det['class']})")
                    break
            
            if not is_duplicate:
                filtered_detections.append(det)
                detection_info.append(det)
        
        print(f"Rilevamenti originali: {len(all_detections)}, Dopo filtro: {len(filtered_detections)}")
        
        for det in filtered_detections:
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            color = self.class_colors.get(class_name, (192, 192, 192))  # Grigio come default
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Testo bianco
        
        # Aggiungi una legenda in basso a destra dell'immagine
        img_height, img_width = image_rgb.shape[:2]
        legend_y = img_height - 10
        cv2.putText(result_image, 
                   f"Correzioni applicate: {self.corrections_applied}", 
                   (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, 
                   (255, 255, 255), 
                   1, 
                   cv2.LINE_AA,
                   False)
        
        os.remove(temp_path)
        
        report = self.create_report(detection_info, confidence_threshold)
        
        # Aggiungi info sulle correzioni al report
        if self.anatomical_correction_enabled:
            report += f"\nCorrezioni anatomiche applicate: {self.corrections_applied}\n"
            if self.corrections_applied > 0:
                report += "Le etichette sono state corrette in base alla posizione nell'immagine.\n"
        else:
            report += "\nCorrezione anatomica disabilitata.\n"
            
        return (result_image, report)
        def apply_anatomical_correction(detections):
            corrected_detections = []
            
            img_height, img_width = image_rgb.shape[:2]
            img_center_x = img_width / 2
            
            print(f"Centro immagine: x={img_center_x}")
            
            correction_map = {
                'areola_destra': 'areola_sinistra',
                'areola_sinistra': 'areola_destra',
                'seno_destro': 'seno_sinistro',
                'seno_sinistro': 'seno_destro',
                'capezzolo_dx': 'capezzolo_sx',
                'capezzolo_sx': 'capezzolo_dx',
                'giugulare_dx': 'giugulare_sx',
                'giugulare_sx': 'giugulare_dx'
            }
            
            # Aggiungiamo anche la mappa inversa per i nomi delle classi
            class_id_map = {}
            for id, name in self.model.names.items():
                for original, corrected in correction_map.items():
                    if name == original:
                        # Troviamo l'ID della classe corretta
                        for corr_id, corr_name in self.model.names.items():
                            if corr_name == corrected:
                                class_id_map[id] = corr_id
            
            for det in detections:
                # Ottieni le coordinate del box
                x1, y1, x2, y2 = det['bbox']
                
                # Calcola il centro del box rilevato
                box_center_x = (x1 + x2) / 2
                
                # Ottieni il nome e l'ID della classe
                class_name = det['class']
                class_id = det['class_id']
                
                # Verifica se il rilevamento è nel lato destro o sinistro dell'immagine
                on_left_side = box_center_x < img_center_x
                
                # Verifica se è necessaria una correzione
                should_correct = False
                
                # Applica le regole logiche
                if ('destra' in class_name or '_dx' in class_name) and not on_left_side:
                    should_correct = True
                    print(f"Correzione: {class_name} rilevato nel lato destro dell'immagine")
                elif ('sinistra' in class_name or '_sx' in class_name) and on_left_side:
                    should_correct = True
                    print(f"Correzione: {class_name} rilevato nel lato sinistro dell'immagine")
                
                # Se è necessaria una correzione, cambia la classe
                if should_correct and class_name in correction_map:
                    corrected_name = correction_map[class_name]
                    corrected_id = class_id_map.get(class_id, class_id)
                    
                    print(f"Correzione applicata: {class_name} -> {corrected_name}")
                    
                    det_copy = det.copy()
                    det_copy['class'] = corrected_name
                    det_copy['class_id'] = corrected_id
                    corrected_detections.append(det_copy)
                else:
                    corrected_detections.append(det)
            
            return corrected_detections
        
        # Applica la correzione anatomica
        all_detections = apply_anatomical_correction(all_detections)
        
        # Filtra i rilevamenti per evitare sovrapposizioni eccessive
        filtered_detections = []
        for det in all_detections:
            bbox = det['bbox']
            is_duplicate = False
            
            for existing_det in filtered_detections:
                existing_bbox = existing_det['bbox']
                iou = calculate_iou(bbox, existing_bbox)
                
                if iou > 0.5:  # Soglia IoU per considerare due rilevamenti come duplicati
                    # È un duplicato, lo saltiamo
                    is_duplicate = True
                    print(f"Filtrato rilevamento duplicato: {det['class']} (sovrapposto con {existing_det['class']})")
                    break
            
            if not is_duplicate:
                filtered_detections.append(det)
                detection_info.append(det)
        
        print(f"Rilevamenti originali: {len(all_detections)}, Dopo filtro: {len(filtered_detections)}")
        
        # Ora disegniamo solo i rilevamenti filtrati
        for det in filtered_detections:
            class_name = det['class']
            conf = det['confidence']
            x1, y1, x2, y2 = det['bbox']
            
            # Usa il colore corrispondente alla classe o un colore di default
            color = self.class_colors.get(class_name, (192, 192, 192))  # Grigio come default
            
            # Disegna il rettangolo con il colore appropriato
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Disegna l'etichetta con sfondo per migliorare la leggibilità
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Testo bianco
        
        os.remove(temp_path)
        
        report = self.create_report(detection_info, confidence_threshold)
        
        # Converti l'immagine al formato corretto per l'output
        return (result_image, report)
    
    def create_report(self, detections, confidence_threshold):
        """Crea un report testuale dei rilevamenti"""
        if not detections:
            return "Nessun oggetto rilevato. Prova a diminuire la soglia di confidenza."
        
        report = f"REPORT RILEVAMENTO - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += "=" * 50 + "\n\n"
        report += f"Soglia di confidenza: {confidence_threshold}\n"
        report += f"Oggetti rilevati: {len(detections)}\n\n"
        
        # Raggruppa per classe
        by_class = {}
        for det in detections:
            class_name = det['class']
            if class_name not in by_class:
                by_class[class_name] = []
            by_class[class_name].append(det)
        
        # Riporta per classe, ordinando per nomi di classe
        for class_name in sorted(by_class.keys()):
            items = by_class[class_name]
            report += f"{class_name.upper()} ({len(items)} rilevamenti):\n"
            for i, det in enumerate(items, 1):
                x1, y1, x2, y2 = det['bbox']
                width = x2 - x1
                height = y2 - y1
                area = width * height
                conf = det['confidence']
                
                # Calcola la posizione rispetto al centro dell'immagine
                center_x = (x1 + x2) / 2
                position = "sinistra" if center_x < 320 else "destra"  # Valore approssimativo
                
                report += f"{i}. Confidenza: {conf:.2f}\n"
                report += f"   Dimensione: {width}x{height} px (area: {area} px²)\n"
                report += f"   Posizione: lato {position} dell'immagine\n"
                report += f"   Coordinate: ({x1}, {y1}) - ({x2}, {y2})\n\n"
        
        # Informazioni aggiuntive
        report += "INFORMAZIONI AGGIUNTIVE:\n"
        report += "- La logica di correzione anatomica è stata applicata per garantire etichette corrette.\n"
        report += "- Le strutture 'destre' dovrebbero apparire sul lato sinistro dell'immagine (punto di vista dell'osservatore).\n"
        report += "- Le strutture 'sinistre' dovrebbero apparire sul lato destro dell'immagine.\n"
        
        return report
    
    def save_results(self, image, report):
        """Salva i risultati dell'inferenza"""
        if image is None:
            return "❌ Nessuna immagine da salvare."
        
        # Cartella per i risultati
        results_dir = "detection_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Timestamp per nomi file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Salva immagine
        image_path = os.path.join(results_dir, f"detection_{timestamp}.jpg")
        # Assicurati che l'immagine sia nel formato corretto prima del salvataggio
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            cv2.imwrite(image_path, image)
        else:
            return "❌ Formato immagine non valido per il salvataggio."
        
        # Salva report
        report_path = os.path.join(results_dir, f"report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        
        return f"✅ Risultati salvati:\n- Immagine: {image_path}\n- Report: {report_path}"


def create_gui():
    """Crea l'interfaccia grafica Gradio"""
    inference_engine = MedicalYOLOInference()
    
    with gr.Blocks(title="Rilevamento Anatomico con YOLO") as demo:
        gr.Markdown("""
        # 🔍 Rilevamento Anatomico con YOLO
        
        Questo strumento permette di analizzare immagini per il rilevamento di strutture anatomiche
        in pazienti con mastopessi.
        
        ## Istruzioni:
        1. Carica il modello YOLO (.pt)
        2. Carica un'immagine da analizzare
        3. Regola la soglia di confidenza (0-1)
        4. Clicca "Analizza"
        5. Salva i risultati se necessario
        """)
        
        with gr.Row():
            with gr.Column():
                model_path = gr.File(label="Carica Modello YOLO (.pt)", file_types=[".pt"])
                load_model_btn = gr.Button("Carica Modello")
                model_status = gr.Textbox(label="Stato Modello", interactive=False)
                
                input_image = gr.Image(label="Immagine da Analizzare", type="numpy")
                confidence = gr.Slider(
                    minimum=0.1, 
                    maximum=0.9, 
                    value=0.5, 
                    step=0.05, 
                    label="Soglia di Confidenza"
                )
                anatomical_correction = gr.Checkbox(
                    label="Abilita Correzione Anatomica",
                    value=True,
                    info="Corregge automaticamente le etichette in base alla posizione nell'immagine"
                )
                analyze_btn = gr.Button("Analizza Immagine", variant="primary")
            
            with gr.Column():
                output_image = gr.Image(label="Risultato", type="numpy")
                output_report = gr.Textbox(label="Report", lines=20)
                save_btn = gr.Button("Salva Risultati")
                save_status = gr.Textbox(label="Stato Salvataggio", interactive=False)
        
        # Funzione di caricamento modello
        def load_model_fn(model_file):
            if model_file is None:
                return "❌ Nessun file selezionato."
            return inference_engine.load_model(model_file.name)
        
        # Funzione di analisi con controllo correzione anatomica
        def analyze_with_correction(image, confidence, anatomical_correction, model_path):
            inference_engine.anatomical_correction_enabled = anatomical_correction
            return inference_engine.analyze_image(image, confidence, model_path)
        
        # Connetti funzioni
        load_model_btn.click(
            fn=load_model_fn,
            inputs=[model_path],
            outputs=[model_status]
        )
        
        analyze_btn.click(
            fn=analyze_with_correction,
            inputs=[input_image, confidence, anatomical_correction, model_path],
            outputs=[output_image, output_report]
        )
        
        save_btn.click(
            fn=inference_engine.save_results,
            inputs=[output_image, output_report],
            outputs=[save_status]
        )
    
    return demo


if __name__ == "__main__":
    demo = create_gui()
    demo.launch(share=False)