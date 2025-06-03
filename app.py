import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import requests
import zipfile
import io
import shutil

# Definizione del modello U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.down5 = DoubleConv(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(128, 64)
        
        self.outconv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Percorso discendente
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x5 = self.down5(self.pool(x4))
        
        # Percorso ascendente
        x = self.up1(x5)
        x = self.upconv1(torch.cat([x4, x], dim=1))
        
        x = self.up2(x)
        x = self.upconv2(torch.cat([x3, x], dim=1))
        
        x = self.up3(x)
        x = self.upconv3(torch.cat([x2, x], dim=1))
        
        x = self.up4(x)
        x = self.upconv4(torch.cat([x1, x], dim=1))
        
        x = self.outconv(x)
        x = self.sigmoid(x)
        
        return x

# Dataset personalizzato per le immagini mammografiche
class MammographyDataset(Dataset):
    def __init__(self, img_paths, mask_paths, transform=None):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # Caricamento dell'immagine
        image = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

# Funzione per caricare i dati
def load_data(img_dir, mask_dir, batch_size=8, val_split=0.2, test_split=0.1):
    img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')])
    mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.png') or f.endswith('.jpg')])
    
    # Split dei dati in train, validation e test
    train_img, temp_img, train_mask, temp_mask = train_test_split(
        img_paths, mask_paths, test_size=val_split+test_split, random_state=42
    )
    
    val_img, test_img, val_mask, test_mask = train_test_split(
        temp_img, temp_mask, test_size=test_split/(val_split+test_split), random_state=42
    )
    
    # Trasformazioni
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # Creazione dei dataset
    train_dataset = MammographyDataset(train_img, train_mask, transform)
    val_dataset = MammographyDataset(val_img, val_mask, transform)
    test_dataset = MammographyDataset(test_img, test_mask, transform)
    
    # Creazione dei dataloader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

# Funzione per addestrare il modello
def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Definizione della loss e dell'ottimizzatore
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
            
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Stampa dei risultati
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # Salvataggio del modello
    torch.save(model.state_dict(), "mammography_segmentation_model.pth")
    
    # Plot della curva di apprendimento
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('learning_curve.png')
    plt.close()
    
    return model, train_losses, val_losses

# Funzione per valutare il modello
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Metriche di valutazione
    dice_scores = []
    iou_scores = []
    
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            
            # Calcolo del Dice score (coefficiente di somiglianza)
            pred = (outputs > 0.5).float()
            intersection = (pred * masks).sum((1, 2, 3))
            union = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3))
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice.cpu().numpy())
            
            # Calcolo dell'IoU (Intersection over Union)
            intersection = (pred * masks).sum((1, 2, 3))
            union = pred.sum((1, 2, 3)) + masks.sum((1, 2, 3)) - intersection
            iou = intersection / (union + 1e-8)
            iou_scores.append(iou.cpu().numpy())
    
    dice_scores = np.concatenate(dice_scores)
    iou_scores = np.concatenate(iou_scores)
    
    mean_dice = np.mean(dice_scores)
    mean_iou = np.mean(iou_scores)
    
    print(f"Mean Dice Score: {mean_dice:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    return mean_dice, mean_iou

# Funzione per la predizione su nuove immagini
def predict(model, image_path, transform=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Caricamento e preparazione dell'immagine
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predizione
    with torch.no_grad():
        output = model(image_tensor)
        prediction = (output > 0.5).float()
    
    # Conversione a immagine
    prediction = prediction.squeeze().cpu().numpy()
    prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))
    prediction_img = prediction_img.resize(original_size)
    
    # Visualizzazione
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Immagine originale")
    plt.subplot(1, 2, 2)
    plt.imshow(prediction_img, cmap='gray')
    plt.title("Segmentazione predetta")
    plt.savefig('prediction_result.png')
    plt.close()
    
    return prediction_img

# Esempio di utilizzo
if __name__ == "__main__":
    # Directory dei dati
    dataset_dir = "mias-mammography"
    
    # Crea la directory del dataset se non esiste
    os.makedirs(dataset_dir, exist_ok=True)
    
    # URL per il download del dataset (immagini e maschere)
    # Utilizziamo il dataset CBIS-DDSM che contiene sia immagini che maschere
    dataset_url = "https://s3.amazonaws.com/mammo-data-download/sample_images.zip"  # URL di esempio
    masks_url = "https://s3.amazonaws.com/mammo-data-download/sample_masks.zip"    # URL di esempio per le maschere
    
    # Scarica e estrai il dataset di immagini
    print("Scaricamento delle immagini mammografiche...")
    try:
        response = requests.get(dataset_url, stream=True)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                print("Estrazione delle immagini...")
                zip_ref.extractall(os.path.join(dataset_dir, "temp_images"))
            print("Immagini scaricate e estratte con successo.")
        else:
            print(f"Errore durante il download delle immagini: {response.status_code}")
            print("URL di esempio fornito. Sostituisci con un URL valido o scarica manualmente le immagini.")
    except Exception as e:
        print(f"Errore durante il download o l'estrazione delle immagini: {e}")
    
    # Scarica e estrai le maschere
    print("Scaricamento delle maschere...")
    try:
        response = requests.get(masks_url, stream=True)
        if response.status_code == 200:
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                print("Estrazione delle maschere...")
                zip_ref.extractall(os.path.join(dataset_dir, "temp_masks"))
            print("Maschere scaricate e estratte con successo.")
        else:
            print(f"Errore durante il download delle maschere: {response.status_code}")
            print("URL di esempio fornito. Sostituisci con un URL valido o scarica manualmente le maschere.")
    except Exception as e:
        print(f"Errore durante il download o l'estrazione delle maschere: {e}")
    
    # Organizza le immagini e le maschere nelle directory corrette
    img_dir = os.path.join(dataset_dir, "images")
    mask_dir = os.path.join(dataset_dir, "masks")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    # Trova le immagini e le maschere scaricate
    temp_img_dir = os.path.join(dataset_dir, "temp_images")
    temp_mask_dir = os.path.join(dataset_dir, "temp_masks")
    
    # Funzione per cercare file con estensioni specifiche
    def find_images(directory, extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        images = []
        if os.path.exists(directory):
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(extensions):
                        images.append(os.path.join(root, file))
        return images
    
    # Trova le immagini e le maschere
    images = find_images(temp_img_dir)
    masks = find_images(temp_mask_dir)
    
    print(f"Trovate {len(images)} immagini e {len(masks)} maschere.")
    
    # Se non ci sono abbastanza immagini o maschere, offri alternative
    if len(images) < 5 or len(masks) < 5:
        print("Il numero di immagini o maschere trovate è insufficiente.")
        print("Utilizziamo un dataset alternativo: MINI-MIAS (Mammographic Image Analysis Society)")
        
        mini_mias_url = "http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz"
        print(f"Scaricamento del dataset MINI-MIAS da {mini_mias_url}...")
        try:
            response = requests.get(mini_mias_url, stream=True)
            if response.status_code == 200:
                # Salva il file tar.gz
                tar_path = os.path.join(dataset_dir, "all-mias.tar.gz")
                with open(tar_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Estrai il file tar.gz (usando os.system per gestire file tar.gz)
                import tarfile
                with tarfile.open(tar_path) as tar:
                    tar.extractall(path=os.path.join(dataset_dir, "mini-mias"))
                
                print("Dataset MINI-MIAS scaricato e estratto con successo.")
                
                # Ora dobbiamo generare le maschere dall'annotazione MINI-MIAS
                # MINI-MIAS fornisce coordinate e raggi delle anomalie in un file info
                
                # Per semplicità, creiamo alcune maschere di esempio (in un caso reale, useremmo i dati di annotazione)
                mini_mias_dir = os.path.join(dataset_dir, "mini-mias")
                mini_mias_images = find_images(mini_mias_dir)
                
                # Svuota e ricrea le directory di destinazione
                shutil.rmtree(img_dir, ignore_errors=True)
                shutil.rmtree(mask_dir, ignore_errors=True)
                os.makedirs(img_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                
                print("Copiando le immagini nella directory di destinazione...")
                for i, img_path in enumerate(mini_mias_images[:30]):  # Limitiamo a 30 immagini per esempio
                    # Copia l'immagine originale
                    img_filename = f"image_{i+1:03d}.png"
                    img = Image.open(img_path).convert("RGB")
                    img.save(os.path.join(img_dir, img_filename))
                    
                    # Crea una maschera fittizia (in un caso reale useremmo le annotazioni vere)
                    mask = Image.new('L', img.size, 0)  # Maschera nera
                    draw = Image.new('L', img.size, 0)
                    
                    # Disegna un cerchio bianco in una posizione casuale per simulare il tessuto mammario
                    import random
                    from PIL import ImageDraw
                    
                    width, height = img.size
                    x_center = random.randint(width // 3, width * 2 // 3)
                    y_center = random.randint(height // 3, height * 2 // 3)
                    radius = min(width, height) // 4
                    
                    draw = ImageDraw.Draw(mask)
                    draw.ellipse((x_center - radius, y_center - radius, 
                                 x_center + radius, y_center + radius), fill=255)
                    
                    mask_filename = f"mask_{i+1:03d}.png"
                    mask.save(os.path.join(mask_dir, mask_filename))
                
                print(f"Create {min(30, len(mini_mias_images))} immagini e maschere.")
                
            else:
                print(f"Errore durante il download del dataset MINI-MIAS: {response.status_code}")
        except Exception as e:
            print(f"Errore durante il download o l'estrazione del dataset MINI-MIAS: {e}")
    else:
        # Copia le immagini e le maschere nelle directory corrette
        print("Copiando le immagini e le maschere nelle directory di destinazione...")
        for i, (img_path, mask_path) in enumerate(zip(images[:min(len(images), len(masks))], masks[:min(len(images), len(masks))])):
            # Copia l'immagine
            img_filename = f"image_{i+1:03d}.png"
            shutil.copy(img_path, os.path.join(img_dir, img_filename))
            
            # Copia la maschera
            mask_filename = f"mask_{i+1:03d}.png"
            shutil.copy(mask_path, os.path.join(mask_dir, mask_filename))
        
        print(f"Copiate {min(len(images), len(masks))} immagini e maschere.")
    
    # Pulisci le directory temporanee
    shutil.rmtree(temp_img_dir, ignore_errors=True)
    shutil.rmtree(temp_mask_dir, ignore_errors=True)
    
    # Caricamento dei dati
    train_loader, val_loader, test_loader = load_data(img_dir, mask_dir, batch_size=8)
    
    # Inizializzazione del modello
    model = UNet(in_channels=3, out_channels=1)
    
    # Verificare che ci siano immagini e maschere
    if not os.listdir(img_dir) or not os.listdir(mask_dir):
        print("Le cartelle delle immagini o delle maschere sono vuote.")
        exit(1)
    
    print(f"Trovate {len(os.listdir(img_dir))} immagini e {len(os.listdir(mask_dir))} maschere.")
    
    # Caricamento dei dati
    train_loader, val_loader, test_loader = load_data(img_dir, mask_dir, batch_size=8)
    
    # Inizializzazione del modello
    model = UNet(in_channels=3, out_channels=1)
    
    # Addestramento del modello
    model, train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=50)
    
    # Valutazione del modello
    mean_dice, mean_iou = evaluate_model(model, test_loader)
    
    # Predizione su una nuova immagine (usa la prima immagine del test set)
    test_images = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    if test_images:
        test_image_path = test_images[0]
        print(f"Effettuando predizione su: {test_image_path}")
        prediction = predict(model, test_image_path)
    else:
        print("Nessuna immagine trovata per la predizione.")