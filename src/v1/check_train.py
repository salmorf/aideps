# check_training_quality.py
from ultralytics import YOLO
import matplotlib.pyplot as plt
from pathlib import Path

def check_training_metrics():
    """Verifica le metriche del training"""
    # Percorso ai risultati del training
    results_path = Path('runs/train/breast_detection_m4_optimized')
    
    # Carica e mostra i grafici del training
    plots = ['confusion_matrix.png', 'results.png', 'PR_curve.png', 'F1_curve.png']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, plot_name in enumerate(plots):
        plot_path = results_path / plot_name
        if plot_path.exists():
            img = plt.imread(str(plot_path))
            axes[i].imshow(img)
            axes[i].set_title(plot_name)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def evaluate_model_details(model_path):
    """Valutazione dettagliata del modello"""
    model = YOLO(model_path)
    
    # Valuta sul validation set
    metrics = model.val()
    
    print("Metriche dettagliate:")
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")  
    print(f"Recall: {metrics.box.mr:.3f}")
    
    return metrics

# Esegui
if __name__ == "__main__":
    check_training_metrics()
    model_path = 'runs/train/breast_detection_m4_optimized/weights/best.pt'
    evaluate_model_details(model_path)