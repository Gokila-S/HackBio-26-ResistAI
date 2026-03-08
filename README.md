# ResistAI — AI-Powered Antibiotic Resistance Predictor from Genomics

> 🧬 Genotype-to-phenotype prediction of antibiotic resistance in *Escherichia coli* using Machine Learning

## 🎯 Overview

ResistAI is an advanced machine learning platform that predicts antibiotic resistance from bacterial genomic data. Given a genome assembly in FASTA format, it extracts AMR genomic features and predicts resistance phenotypes (Resistant, Susceptible, Intermediate) for three clinically important antibiotics — with clinical decision support, WHO AWaRe classification, and interactive visualizations.

**Target Organism:** *Escherichia coli* (Gram-negative)  
**Target Antibiotics:** Ampicillin, Ciprofloxacin, Gentamicin

## 🚀 Key Features

### 🔬 Core ML Pipeline
- **FASTA Input:** Upload or paste bacterial genome assemblies
- **Multi-drug Prediction:** Simultaneous resistance prediction for 3 antibiotics
- **Confidence Scores:** Probability distributions for each prediction
- **Feature Importance (SHAP):** Biological interpretability of key resistance markers

### 🏥 Clinical Decision Support
- **Clinical Recommendations:** Actionable treatment guidance per antibiotic
- **WHO AWaRe Classification:** Access/Watch/Reserve categorization for antibiotic stewardship
- **Antibiotic Stewardship Score:** Quantified stewardship metric (0-100)
- **MDR Detection:** Automatic multi-drug resistance flagging with specialist referral

### 📊 Advanced Visualizations
- **Gene Network Visualization:** Interactive D3.js force-directed graph showing resistance gene relationships
- **Resistance Heatmap:** Gene-antibiotic correlation heatmap with intensity mapping
- **Resistance Evolution Timeline:** Step-by-step visualization of how resistance accumulates
- **Animated DNA Helix:** Background visualization with molecular aesthetics

### 🧬 Molecular Biology
- **Resistance Mechanism Explainer:** Detailed molecular mechanism descriptions for each detected gene (enzymatic hydrolysis, target modification, efflux pumps, etc.)
- **Gene Panel Database:** 51 curated AMR genes across 3 mechanism categories (primary, secondary, SNPs)
- **18+ Detailed Mechanism Profiles:** Full molecular descriptions including location, prevalence, and molecular targets

### 📊 Batch & Epidemiological Analysis
- **Batch Genome Analysis:** Analyze up to 10 isolates simultaneously
- **Epidemiological Clustering:** Automatic grouping by resistance profile
- **MDR Prevalence Tracking:** Batch-level MDR statistics

### 💾 Additional Features
- **Analysis History:** Local storage-based history of past analyses
- **Clinical Report Export:** Downloadable comprehensive resistance reports
- **File Upload Support:** Direct FASTA file upload
- **Real-time Sequence Stats:** GC content, contig count, sequence length
- **Particle Celebration Effects:** Visual feedback on successful analysis
- **Responsive Design:** Works on desktop, tablet, and mobile

## 📈 Model Performance

| Antibiotic     | Accuracy | AUC    | MCC   |
|----------------|----------|--------|-------|
| Ampicillin     | 91.5%    | 98.9%  | 0.862 |
| Ciprofloxacin  | 92.3%    | 98.6%  | 0.874 |
| Gentamicin     | 94.3%    | 98.8%  | 0.906 |

## 🛠️ Tech Stack

- **ML:** XGBoost, scikit-learn, SHAP
- **Backend:** Flask (Python)
- **Frontend:** HTML5, CSS3 (Glassmorphism), JavaScript, D3.js, Chart.js
- **Data Source:** BV-BRC (PATRIC) AMR Database
- **Design:** Premium dark UI with glassmorphism, micro-animations, animated gradients

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train models
python train_model.py

# Launch web app
python app.py
```

Then visit **http://127.0.0.1:5000**

## 📁 Project Structure

```
hackbio-iit-mandi/
├── app.py                  # Flask web server with all API endpoints
├── train_model.py          # ML training pipeline
├── requirements.txt        # Python dependencies
├── clinical_guide.md       # Clinical interpretation guide
├── static/
│   ├── style.css           # Premium glassmorphism UI styles
│   └── app.js              # Frontend application logic
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

## 🏆 Judging Criteria Mapping

| Criteria                    | Weight | Our Implementation |
|-----------------------------|--------|---------------------|
| Predictive Performance      | 35%    | XGBoost with AUC >98%, MCC >0.86 |
| Feature Interpretability    | 25%    | SHAP values + Gene Network + Heatmap + Mechanism Explainer |
| Clinical Actionability      | 25%    | Treatment recommendations + WHO AWaRe + Stewardship Score |
| Runtime Efficiency          | 15%    | <2 second prediction time |

## 🧬 Unique Differentiators

1. **WHO AWaRe Integration** — First AMR tool to integrate WHO's Access/Watch/Reserve classification directly into predictions
2. **Resistance Evolution Timeline** — Visualize step-by-step how resistance develops through gene acquisition  
3. **Molecular Mechanism Explainer** — Click any gene to understand *how* it causes resistance at the molecular level
4. **Epidemiological Batch Analysis** — Analyze multiple isolates with automatic clustering by resistance profile
5. **Antibiotic Stewardship Score** — Quantified metric for antibiotic prescribing quality

## ⚠️ Disclaimer

This tool is for **RESEARCH and DECISION SUPPORT only**. Always confirm predictions with phenotypic antimicrobial susceptibility testing (AST).

## 👥 Team

Built at **HackBio IIT Mandi** — Organised by Kamand Bioengineering Group
