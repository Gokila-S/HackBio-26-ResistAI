# ResistAI — Antibiotic Resistance Predictor from Genomics

> 🧬 AI-powered genotype-to-phenotype prediction of antibiotic resistance in *Escherichia coli*

## Overview

ResistAI is a machine learning tool that predicts antibiotic resistance from bacterial genomic data. Given a genome assembly in FASTA format, it extracts AMR genomic features and predicts resistance phenotypes (Resistant, Susceptible, Intermediate) for three clinically important antibiotics.

**Target Organism:** *Escherichia coli* (Gram-negative)  
**Target Antibiotics:** Ampicillin, Ciprofloxacin, Gentamicin

## Features

- **FASTA Input:** Upload or paste bacterial genome assemblies
- **Multi-drug Prediction:** Simultaneous resistance prediction for 3 antibiotics
- **Confidence Scores:** Probability distributions for each prediction
- **Clinical Recommendations:** Actionable treatment guidance
- **Gene Network Visualization:** Interactive D3.js force-directed graph showing resistance gene relationships
- **Feature Importance (SHAP):** Biological interpretability of key resistance markers
- **Real-time Analysis:** Fast inference pipeline

## Model Performance

| Antibiotic     | Accuracy | AUC    | MCC   |
|----------------|----------|--------|-------|
| Ampicillin     | 91.5%    | 98.9%  | 0.862 |
| Ciprofloxacin  | 92.3%    | 98.6%  | 0.874 |
| Gentamicin     | 94.3%    | 98.8%  | 0.906 |

## Tech Stack

- **ML:** XGBoost, scikit-learn, SHAP
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3, JavaScript, D3.js, Chart.js
- **Data Source:** BV-BRC (PATRIC) AMR Database

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_model.py

# Launch web app
python app.py
```

Then visit **http://127.0.0.1:5000**

## Project Structure

```
hackbio-iit-mandi/
├── app.py                  # Flask web server
├── train_model.py          # ML training pipeline
├── requirements.txt        # Python dependencies
├── clinical_guide.md       # Clinical interpretation guide
├── templates/
│   └── index.html          # Web interface
├── models/
│   ├── model_ampicillin.joblib
│   ├── model_ciprofloxacin.joblib
│   ├── model_gentamicin.joblib
│   ├── metadata.json
│   └── shap_values.json
└── data/
    ├── patric_ecoli_amr.csv
    └── amr_genomic_features.csv
```

## Judging Criteria Mapping

| Criteria                    | Weight | Our Implementation |
|-----------------------------|--------|--------------------|
| Predictive Performance      | 35%    | XGBoost with AUC >98%, MCC >0.86 |
| Feature Interpretability    | 25%    | SHAP values + Gene Network Viz |
| Clinical Actionability      | 25%    | Treatment recommendation engine |
| Runtime Efficiency          | 15%    | <2 second prediction time |

## Team

Built at **HackBio IIT Mandi** — Organised by Kamand Bioengineering Group
