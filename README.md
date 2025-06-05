# AIDEPS

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) [![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/) [![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO-blue)](https://ultralytics.com/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An advanced medical object detection system based on YOLOv8 with innovative AI, uncertainty quantification, explainable AI and auto-tuning capabilities.

## WebApplication Demo
The Aideps Demo Live is available at http://aideps.kazaamlab.com:3000/. You can test the application directly here.

To access the demo, use the following credentials:

Username: demo@demo.com
Password: demo

## Backend Technologies Used

- **FastAPI** – Python framework for building RESTful APIs  
- **MongoDB** – NoSQL database for storing user and dataset information  
- **JWT (JSON Web Token)** – Secure token-based authentication  
- **Pandas** – Data manipulation and dataset handling  
- **Scikit-learn** – Machine Learning library  
- **Motor** – Asynchronous driver for MongoDB  
- **Python-Decouple** – Environment variable management via `.env` files  
- **Uvicorn** – ASGI server for running FastAPI applications

## Frontend Technologies Used

- **Next.js** – React framework for server-side rendering and full-stack capabilities  
- **React** – JavaScript library for building user interfaces  
- **TypeScript** – Strongly typed superset of JavaScript for safer and scalable development  
- **Tailwind CSS** – Utility-first CSS framework for fast and responsive UI design  
- **Turbopack** – High-performance bundler and build system for Next.js applications

## Source Code

- [Backend (FastAPI)](https://github.com/salmorf/aideps-be)
- [Frontend (Next.js)](https://github.com/salmorf/aideps-fe)
- [Feature Based Model](https://github.com/VCJx07/CatBoost-Mastopexy-Feature-Based-Model/tree/master)


### Installazione con Requirements
```bash
git clone <repository-url>
cd aideps
pip install -r requirements.txt
```

### 1.Dataset Preparation
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


### Installation with Requirements
```bash
git clone <repository-url>
cd aideps
pip install -r requirements.txt
```

### 1. Dataset Initialization
```python
# Required directory structure:
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

nc: 8 
names: ['classe1', 'classe2', 'classe3', 'classe4', 'classe5', 'classe6', 'classe7', 'classe8']
```

### Execution
```
python src/v2/train_medical_yolo.py    
```

### File Generati
```
comprehensive_report_YYYYMMDD_HHMMSS/
├── comprehensive_report.json         # Structured Data Report
├── comprehensive_medical_yolo_report.pdf  # Full PDF report
├── ensemble_performance.png           # Performance Ensemble Graphs
├── uncertainty_analysis.png           # Uncertainty analysis
├── feature_importance_radar.png       # Importanza features
└── optimization_convergence.png       # Convergenza ottimizzazione
```

### Key Metrics
```
- **mAP@0.5**: Mean Average Precision a soglia IoU 0.5
- **mAP@0.5:0.95**: Mean Average Precision media su IoU 0.5-0.95
- **Uncertainty Score**: Misura dell'incertezza dell'ensemble
- **Diversity Score**: Diversità dei modelli nell'ensemble
- **Calibration Score**: Calibrazione delle confidence
```

### Technical Limitations
```
- **Memory intensive**: Ensemble richiede molta GPU memory
- **Training time**: Processo completo può richiedere molte ore
- **Dependencies**: Molte dipendenze per funzionalità complete
```

### Medical Considerations
- ⚠️ **NON per uso diagnostico diretto** senza validazione clinica
- ⚠️ **Sempre validare** con professionisti medici qualificati
- ⚠️ **Compliance normativa** responsabilità dell'utente finale

## 📄 Licenza

Distribuito sotto licenza MIT. Vedi `LICENSE` per maggiori informazioni.
