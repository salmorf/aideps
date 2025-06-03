import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import os

def detect_breast_improved(image_path, debug=False):
    """
    Rilevamento migliorato del seno usando forme circolari/ellittiche
    """

    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine {image_path}")
        return None, None
    
    original = img.copy()
    height, width = img.shape[:2]
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    skin_masks = []
    
    lower_light = np.array([0, 10, 60], dtype=np.uint8)
    upper_light = np.array([20, 150, 255], dtype=np.uint8)
    skin_masks.append(cv2.inRange(hsv, lower_light, upper_light))

    lower_medium = np.array([15, 30, 80], dtype=np.uint8)
    upper_medium = np.array([35, 170, 255], dtype=np.uint8)
    skin_masks.append(cv2.inRange(hsv, lower_medium, upper_medium))

    lower_dark = np.array([0, 10, 30], dtype=np.uint8)
    upper_dark = np.array([25, 140, 200], dtype=np.uint8)
    skin_masks.append(cv2.inRange(hsv, lower_dark, upper_dark))
    
    skin_mask = np.zeros((height, width), dtype=np.uint8)
    for mask in skin_masks:
        skin_mask = cv2.bitwise_or(skin_mask, mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    
    # Trova i contorni
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Nessun contorno trovato")
        return img, None
    
    # Analizza ogni contorno per trovare il seno
    best_candidate = None
    best_score = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Filtra contorni troppo piccoli o troppo grandi
        if area < 0.01 * height * width or area > 0.5 * height * width:
            continue
        
        # Calcola il centro di massa
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Preferisci regioni nella parte centrale-inferiore dell'immagine
        # (dove tipicamente si trova il seno)
        position_score = 0
        if 0.3 * width < cx < 0.7 * width:  # Centro orizzontale
            position_score += 1
        if 0.3 * height < cy < 0.8 * height:  # Parte medio-bassa
            position_score += 1
        
        # Calcola la circolarità
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        # Fit ellisse se possibile
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, (width_e, height_e), angle) = ellipse
            
            # Verifica proporzioni dell'ellisse
            if width_e > 0 and height_e > 0:
                aspect_ratio = min(width_e, height_e) / max(width_e, height_e)
                
                # Score basato su forma e posizione
                shape_score = 0
                if 0.4 < aspect_ratio < 1.0:  # Forma ellittica/circolare
                    shape_score += circularity
                
                # Score totale
                total_score = position_score + shape_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_candidate = contour
                    
                    if debug:
                        print(f"Candidato trovato: position_score={position_score}, "
                              f"shape_score={shape_score:.2f}, total_score={total_score:.2f}")
    
    # Se abbiamo trovato un buon candidato
    if best_candidate is not None:
        # Calcola il rettangolo con margine
        x, y, w, h = cv2.boundingRect(best_candidate)
        
        # Aggiungi margine proporzionale
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        
        x = max(0, x - margin_x)
        y = max(0, y - margin_y)
        w = min(width - x, w + 2 * margin_x)
        h = min(height - y, h + 2 * margin_y)
        
        # Disegna il rettangolo giallo
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)
        
        if debug:
            # Mostra anche il contorno rilevato
            cv2.drawContours(img, [best_candidate], -1, (0, 255, 0), 2)
            
            # Mostra il centro
            M = cv2.moments(best_candidate)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        
        return img, (x, y, w, h)
    else:
        print("Nessun candidato seno trovato - usando il contorno più grande")
        # Fallback: usa il contorno più grande
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Margine più conservativo
        margin = 15
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(width - x, w + 2 * margin)
        h = min(height - y, h + 2 * margin)
        
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 4)
        
        return img, (x, y, w, h)

def add_yellow_rectangle_to_breast(image_path, output_path=None, debug=False):
    """
    Funzione principale per aggiungere un rettangolo giallo al seno
    """
    # Rileva il seno e aggiungi il rettangolo
    result_img, rect_coords = detect_breast_improved(image_path, debug=debug)
    
    if result_img is None:
        return None
    
    # Salva l'immagine
    if output_path is None:
        output_path = 'breast_with_yellow_rectangle.jpg'
    
    cv2.imwrite(output_path, result_img)
    
    # Visualizza il risultato
    plt.figure(figsize=(15, 7))
    
    # Immagine originale
    original = cv2.imread(image_path)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('Immagine Originale')
    plt.axis('off')
    
    # Immagine con rettangolo
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('Immagine con Rettangolo Giallo')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    if rect_coords:
        print(f"Rettangolo posizionato a: x={rect_coords[0]}, y={rect_coords[1]}, "
              f"larghezza={rect_coords[2]}, altezza={rect_coords[3]}")
    
    return output_path

def interactive_rectangle_adjustment(image_path):
    """
    Permette di regolare manualmente la posizione del rettangolo
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Errore: Impossibile caricare l'immagine {image_path}")
        return None
    
    original = img.copy()
    height, width = img.shape[:2]
    
    # Prova prima il rilevamento automatico
    result_img, auto_rect = detect_breast_improved(image_path)
    
    if auto_rect:
        x, y, w, h = auto_rect
        print(f"Rilevamento automatico: x={x}, y={y}, w={w}, h={h}")
    else:
        # Valori di default se il rilevamento fallisce
        x, y, w, h = width//4, height//4, width//2, height//2
    
    # Mostra l'immagine con il rettangolo automatico
    preview = original.copy()
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 255), 4)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
    plt.title('Rilevamento Automatico - Vuoi regolarlo manualmente?')
    plt.axis('off')
    plt.show()
    
    adjust = input("Vuoi regolare manualmente il rettangolo? (s/n): ")
    
    if adjust.lower() == 's':
        print("\nInserisci le nuove coordinate:")
        x = int(input(f"x (attuale: {x}): ") or x)
        y = int(input(f"y (attuale: {y}): ") or y)
        w = int(input(f"larghezza (attuale: {w}): ") or w)
        h = int(input(f"altezza (attuale: {h}): ") or h)
    
    # Disegna il rettangolo finale
    final_img = original.copy()
    cv2.rectangle(final_img, (x, y), (x + w, y + h), (0, 255, 255), 4)
    
    # Salva l'immagine
    output_path = 'breast_with_adjusted_rectangle.jpg'
    cv2.imwrite(output_path, final_img)
    
    # Mostra il risultato finale
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
    plt.title('Risultato Finale')
    plt.axis('off')
    plt.show()
    
    print(f"\nImmagine salvata come: {output_path}")
    print(f"Coordinate finali: x={x}, y={y}, w={w}, h={h}")
    
    return output_path

if __name__ == "__main__":
    print("Rilevamento migliorato del seno con rettangolo giallo")
    print("-" * 50)
    
    # Richiedi il percorso dell'immagine
    image_path = input("Inserisci il percorso dell'immagine: ")
    
    if os.path.exists(image_path):
        # Opzione 1: Rilevamento automatico con debug
        print("\n1. Tentativo di rilevamento automatico...")
        result = add_yellow_rectangle_to_breast(image_path, debug=True)
        
        # Opzione 2: Regolazione manuale se necessario
        print("\n2. Regolazione manuale (se necessario)...")
        interactive_result = interactive_rectangle_adjustment(image_path)
    else:
        print(f"File non trovato: {image_path}")