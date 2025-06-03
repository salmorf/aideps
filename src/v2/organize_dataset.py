import os
import shutil

# Percorsi fissi per le directory
RAW_IMAGES_DIR = '/Users/salvatore/Development/KazaamLab/Mastopessi2/data/raw_images'
RAW_LABELS_DIR = '/Users/salvatore/Development/KazaamLab/Mastopessi2/data/raw_labels'
OUTPUT_DIR = '/Users/salvatore/Development/KazaamLab/Mastopessi2/data'

# Classi corrette
CLASS_NAMES =  ['seno_destro', 'seno_sinistro', 'areola_destra', 'areola_sinistra', 'giugulare_dx', 'giugulare_sx', 'capezzolo_dx', 'capezzolo_sx']

def organize_yolo_dataset(
    raw_images_dir=RAW_IMAGES_DIR, 
    raw_labels_dir=RAW_LABELS_DIR, 
    output_dir=OUTPUT_DIR,
    class_names=CLASS_NAMES
):
    """
    Organizza i file immagine e label in una struttura ottimizzata per YOLO.
    
    Args:
    - raw_images_dir (str): Percorso della cartella con le immagini grezze
    - raw_labels_dir (str): Percorso della cartella con le label grezze
    - output_dir (str): Percorso della directory di output per YOLO
    - class_names (list): Nomi delle classi nel dataset
    """
    # Crea le sottocartelle necessarie per YOLO
    train_img_dir = os.path.join(output_dir, 'images', 'train')
    val_img_dir = os.path.join(output_dir, 'images', 'val')
    train_lbl_dir = os.path.join(output_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(output_dir, 'labels', 'val')
    
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)

    # Ottieni la lista dei file
    image_files = set(os.listdir(raw_images_dir))
    label_files = set(os.listdir(raw_labels_dir))

    # Estensioni immagini supportate
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    # Trova i file corrispondenti
    matching_files = []
    for img_file in image_files:
        # Verifica che sia un'immagine
        if os.path.splitext(img_file)[1].lower() not in image_extensions:
            continue

        # Estrai il nome base dell'immagine (senza estensione)
        base_name = os.path.splitext(img_file)[0]
        
        # Controlla se esiste un file label corrispondente
        label_file = base_name + '.txt'
        if label_file in label_files:
            matching_files.append((img_file, label_file))

    # Calcola il numero di file per train e validation
    total_files = len(matching_files)
    train_count = int(total_files * 0.8)  # 80% per training, 20% per validation

    print(f"Trovate {total_files} immagini con label corrispondenti")
    print(f"Training set: {train_count} immagini")
    print(f"Validation set: {total_files - train_count} immagini")

    # Sposta i file
    for i, (img_file, label_file) in enumerate(matching_files):
        # Determina se è un file di training o validazione
        if i < train_count:
            img_dest_dir = train_img_dir
            label_dest_dir = train_lbl_dir
        else:
            img_dest_dir = val_img_dir
            label_dest_dir = val_lbl_dir

        # Copia immagine
        shutil.copy2(
            os.path.join(raw_images_dir, img_file), 
            os.path.join(img_dest_dir, img_file)
        )
        
        # Copia label
        shutil.copy2(
            os.path.join(raw_labels_dir, label_file), 
            os.path.join(label_dest_dir, label_file)
        )

    # Crea il file data.yaml per YOLO
    create_yolo_yaml(output_dir, total_files, class_names)

    print(f"Dataset organizzato con successo in: {output_dir}")

def create_yolo_yaml(output_dir, total_images, class_names):
    """
    Crea un file data.yaml per configurare il dataset YOLO.
    
    Args:
    - output_dir (str): Percorso della directory di output
    - total_images (int): Numero totale di immagini
    - class_names (list): Nomi delle classi
    """
    # Crea il percorso per il file data.yaml nella directory del dataset
    data_yaml_path = os.path.join(os.path.dirname(output_dir), 'data.yaml')
    
    yaml_content = f"""# Configurazione dataset per rilevamento anatomico

# Percorso base del dataset (directory principale)
path: {output_dir}

# Sottocartelle per training e validation
train: images/train
val: images/val

# Numero di classi
nc: {len(class_names)}

# Nomi delle classi (nell'ordine degli ID)
names: {class_names}

# Numero totale di immagini
total_images: {total_images}
"""
    
    with open(data_yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    
    print(f"File data.yaml creato in: {data_yaml_path}")

def main():
    organize_yolo_dataset()

if __name__ == '__main__':
    main()
