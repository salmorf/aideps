# AIDEPS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-blue)](https://ultralytics.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced medical object detection system based on YOLOv8 with innovative AI, uncertainty quantification, explainable AI and auto-tuning capabilities.

## WebApplication Demo
The Aideps Demo Live is available at http://aideps.kazaamlab.com:3000/. You can test the application directly here.

To access the demo, use the following credentials:

Username: demo@demo.com
Password: demo


### Technologies Used
Frontend Nextjs 






<!-- 
## 📦 Installazione

### Requisiti Base
```bash
pip install ultralytics torch torchvision opencv-python numpy matplotlib seaborn scikit-learn pyyaml
```

### Requisiti Completi (Funzionalità Avanzate)
```bash
# Core ML packages
pip install ultralytics torch torchvision

# Computer vision and data processing
pip install opencv-python numpy matplotlib seaborn scikit-learn
pip install albumentations pillow

# Auto-optimization and experiment tracking
pip install optuna wandb mlflow

# Explainable AI
pip install grad-cam

# Monitoring and system info
pip install psutil

# Report generation
pip install reportlab plotly

# Configuration
pip install pyyaml
```

### Installazione con Requirements
```bash
git clone <repository-url>
cd innovative-medical-yolo
pip install -r requirements.txt
```

## 🛠️ Setup Rapido

### 1. Preparazione Dataset
```python
# Struttura directory richiesta:
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### 2. Configurazione data.yaml
```yaml
path: /path/to/dataset
train: train
val: val
test: test

nc: 3  # numero classi
names: ['classe1', 'classe2', 'classe3']
```

### 3. Configurazione Sistema (Opzionale)
```yaml
# innovative_config.yaml
model:
  size: 'l'  # n, s, m, l, x
  input_size: 640
  ensemble_size: 3

training:
  epochs: 200
  batch_size: 16
  auto_tune: true
  use_synthetic_data: true
  uncertainty_quantification: true

optimization:
  use_optuna: true
  n_trials: 50
  pruning: true

explainability:
  enable_gradcam: true
  generate_heatmaps: true
  feature_importance: true

monitoring:
  real_time: true
  log_frequency: 10
  save_checkpoints: true
```

## 🎯 Utilizzo

### Sistema Completo Automatico
```python
from innovative_medical_yolo_ai_enhanced import InnovativeMedicalYOLO

# Inizializzazione
sistema = InnovativeMedicalYOLO("innovative_config.yaml")

# Auto-ottimizzazione
parametri_ottimali = sistema.auto_optimize_hyperparameters("data.yaml")

# Generazione dati sintetici
sistema.generate_synthetic_data("data.yaml", target_augmentation=500)

# Training con uncertainty quantification
modelli_ensemble = sistema.train_with_uncertainty_quantification("data.yaml")

# Setup explainable AI
sistema.setup_explainable_ai()

# Genera report completo
report = sistema.generate_comprehensive_report("data.yaml")
```

### Predizione Avanzata
```python
# Predizione con uncertainty e spiegazioni
risultato = sistema.predict_with_uncertainty_and_explanation(
    "immagine_test.jpg",
    save_explanations=True
)

print(f"Uncertainty media: {risultato['uncertainty']['mean_uncertainty']:.4f}")
print(f"Confidence ensemble: {risultato['prediction']['ensemble_agreement']:.4f}")
print(f"Fattori decisione: {risultato['explanations']['decision_factors']}")
```

### Utilizzo Base (Funzionalità Ridotte)
```python
# Se pacchetti avanzati non disponibili
from innovative_medical_yolo_ai_enhanced import main_innovative

# Avvia con fallback automatico
sistema, report = main_innovative()
```

## 📊 Output e Risultati

### File Generati
```
comprehensive_report_YYYYMMDD_HHMMSS/
├── comprehensive_report.json          # Report dati strutturati
├── comprehensive_medical_yolo_report.pdf  # Report PDF completo
├── ensemble_performance.png           # Grafici performance ensemble
├── uncertainty_analysis.png           # Analisi uncertainty
├── feature_importance_radar.png       # Importanza features
└── optimization_convergence.png       # Convergenza ottimizzazione
```

### Metriche Principali
- **mAP@0.5**: Mean Average Precision a soglia IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision media su IoU 0.5-0.95
- **Uncertainty Score**: Misura dell'incertezza dell'ensemble
- **Diversity Score**: Diversità dei modelli nell'ensemble
- **Calibration Score**: Calibrazione delle confidence

## 🔧 Configurazione Avanzata

### Personalizzazione Hyperparameters
```python
# Parametri personalizzati per ottimizzazione
parametri_custom = {
    'batch_size': 32,
    'lr0': 0.001,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    # ... altri parametri
}

# Training con parametri specifici
sistema.train_with_uncertainty_quantification("data.yaml", parametri_custom)
```

### Monitoring Personalizzato
```python
# Avvia monitoring con intervallo personalizzato
sistema.real_time_monitor.start_monitoring(
    model=modello,
    data_yaml_path="data.yaml",
    interval=30  # secondi
)

# Ottieni summary monitoring
summary = sistema.real_time_monitor.get_monitoring_summary()
```

## 📈 Performance e Benchmarks

### Performance Tipiche
| Configurazione | mAP@0.5 | mAP@0.5:0.95 | Uncertainty | Tempo Training |
|----------------|---------|--------------|-------------|----------------|
| YOLOv8n Ensemble | 0.85-0.90 | 0.65-0.75 | 0.02-0.05 | 2-4h |
| YOLOv8s Ensemble | 0.88-0.93 | 0.70-0.80 | 0.01-0.04 | 4-8h |
| YOLOv8m Ensemble | 0.90-0.95 | 0.75-0.85 | 0.01-0.03 | 8-12h |
| YOLOv8l Ensemble | 0.92-0.97 | 0.80-0.90 | 0.005-0.02 | 12-20h |

### Requisiti Sistema
- **GPU**: NVIDIA GTX 1080 Ti / RTX 2070 o superiore
- **RAM**: 16GB+ raccomandati
- **Storage**: 50GB+ liberi per dataset e modelli
- **Python**: 3.8+ richiesto

## 🔬 Applicazioni Mediche

### Casi d'Uso Validati
- **Radiologia**: Detection lesioni, masse, anomalie
- **Dermatologia**: Classificazione lesioni cutanee
- **Oftalmologia**: Detection patologie retiniche
- **Istologia**: Analisi tessuti e cellule

### Conformità Clinica
- **Uncertainty quantification** per decisioni critiche
- **Explainable AI** per audit e revisione
- **Traceability completa** di training e predizioni
- **Validation metrics** clinicamente rilevanti

## 🤝 Contribuire

### Setup Sviluppo
```bash
git clone <repository-url>
cd innovative-medical-yolo
pip install -e .
pip install -r requirements-dev.txt
```

### Testing
```bash
pytest tests/
python -m pytest tests/ --cov=innovative_medical_yolo
```

### Linee Guida
1. **Code quality**: Utilizzare black, flake8, mypy
2. **Testing**: Coverage minimo 80%
3. **Documentation**: Docstrings Google style
4. **Medical compliance**: Validazione su dataset clinici

## 📚 Documentazione Avanzata

### API Reference
```python
# Documentazione completa classi principali
help(InnovativeMedicalYOLO)
help(MedicalYOLOExplainer)
help(RealTimeMonitor)
```

### Tutorial e Esempi
- [Tutorial Completo](docs/tutorial.md)
- [Esempi Medici](examples/medical_cases/)
- [Best Practices](docs/best_practices.md)
- [Troubleshooting](docs/troubleshooting.md)

## ⚠️ Limitazioni e Considerazioni

### Limitazioni Tecniche
- **Memory intensive**: Ensemble richiede molta GPU memory
- **Training time**: Processo completo può richiedere molte ore
- **Dependencies**: Molte dipendenze per funzionalità complete

### Considerazioni Mediche
- ⚠️ **NON per uso diagnostico diretto** senza validazione clinica
- ⚠️ **Sempre validare** con professionisti medici qualificati
- ⚠️ **Compliance normativa** responsabilità dell'utente finale

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.

## 🙏 Acknowledgments

- **Ultralytics**: Framework YOLOv8 eccellente
- **Optuna**: Optimization framework potente
- **Community**: Contributi e feedback preziosi
- **Medical researchers**: Validazione e use cases

## 📞 Supporto

### Canali di Supporto
- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: support@medical-yolo.com

### FAQ Rapide

**Q: Il sistema funziona senza GPU?**
A: Sì, ma performance molto ridotte. GPU fortemente raccomandata.

**Q: Posso usare i miei dataset personalizzati?**
A: Sì, assicurati formato YOLO e struttura directory corretta.

**Q: Quanto tempo richiede l'ottimizzazione completa?**
A: 4-20 ore a seconda configurazione e dimensione dataset.

**Q: Il sistema è validato per uso clinico?**
A: No, richiede validazione specifica per ogni caso d'uso clinico.

---

<div align="center">

**🏥 Sistema YOLO Medico Innovativo - AI Enhanced**

*Bringing cutting-edge AI to medical imaging with reliability and explainability*

[📚 Documentation](docs/) • [🚀 Examples](examples/) • [🐛 Issues](issues/) • [💬 Discussions](discussions/)

</div> -->