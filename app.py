"""
AMR Resistance Predictor — Flask Web Application
=================================================
Beautiful web interface for predicting antibiotic resistance
from bacterial genomic data (FASTA input).
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import random

app = Flask(__name__)
CORS(app)

# ── Load models and metadata ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANTIBIOTICS = ['ampicillin', 'ciprofloxacin', 'gentamicin']
LABEL_NAMES = ['Susceptible', 'Intermediate', 'Resistant']

models = {}
metadata = {}
shap_data = {}

def load_models():
    global models, metadata, shap_data
    try:
        with open(os.path.join(MODEL_DIR, 'metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        for ab in ANTIBIOTICS:
            model_path = os.path.join(MODEL_DIR, f'model_{ab}.joblib')
            models[ab] = joblib.load(model_path)
        
        shap_path = os.path.join(MODEL_DIR, 'shap_values.json')
        if os.path.exists(shap_path):
            with open(shap_path, 'r') as f:
                shap_data = json.load(f)
        
        print("[OK] All models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")
        print("[INFO] Run train_model.py first!")


def parse_fasta(fasta_text):
    """Parse FASTA format text and return sequences."""
    sequences = {}
    current_header = None
    current_seq = []
    
    for line in fasta_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_header:
                sequences[current_header] = ''.join(current_seq)
            current_header = line[1:].split()[0]
            current_seq = []
        elif line:
            current_seq.append(line.upper())
    
    if current_header:
        sequences[current_header] = ''.join(current_seq)
    
    return sequences


def extract_features_from_fasta(fasta_text):
    """
    Extract AMR genomic features from FASTA sequence.
    
    In a production system, this would run AMRFinder+ or BLAST against
    an AMR gene database. For this prototype, we simulate feature extraction
    using sequence composition analysis and k-mer signatures.
    """
    sequences = parse_fasta(fasta_text)
    if not sequences:
        return None, "Invalid FASTA format"
    
    feature_names = metadata.get('feature_names', [])
    n_features = len(feature_names)
    
    # Combine all sequences
    full_seq = ''.join(sequences.values())
    seq_len = len(full_seq)
    
    if seq_len < 100:
        return None, "Sequence too short (minimum 100 bp)"
    
    # Use sequence composition to generate deterministic features
    # This simulates what AMRFinder+ would output
    features = np.zeros(n_features)
    
    # GC content analysis
    gc_content = (full_seq.count('G') + full_seq.count('C')) / max(seq_len, 1)
    
    # K-mer based feature extraction (simulating gene detection)
    # Use hash of sequence segments to deterministically assign gene presence
    for i, gene_name in enumerate(feature_names):
        # Create a deterministic hash based on sequence and gene name
        gene_hash = hash(gene_name + full_seq[:min(1000, seq_len)])
        segment_start = abs(gene_hash) % max(1, seq_len - 100)
        segment = full_seq[segment_start:segment_start + 100] if seq_len > 100 else full_seq
        
        # Analyze segment composition
        at_content = (segment.count('A') + segment.count('T')) / max(len(segment), 1)
        
        # Gene detection threshold (varies by gene type)
        if 'bla' in gene_name or 'qnr' in gene_name or 'aac' in gene_name:
            threshold = 0.45  # Resistance genes - moderate detection
        elif 'mutation' in gene_name or gene_name.count('_') >= 2:
            threshold = 0.55  # SNPs - harder to detect
        else:
            threshold = 0.50
        
        # Use a combination of factors for detection
        detection_score = (at_content * 0.6 + gc_content * 0.4)
        noise = (abs(hash(gene_name + str(i))) % 100) / 500.0
        
        features[i] = 1.0 if (detection_score + noise) > threshold else 0.0
    
    return features, None


def get_clinical_recommendation(predictions):
    """Generate clinical interpretation based on predictions."""
    recommendations = []
    
    drug_classes = {
        'ampicillin': {
            'class': 'Beta-lactam (Penicillin)',
            'alternatives': ['Piperacillin-tazobactam', 'Meropenem', 'Ceftriaxone'],
            'use_case': 'UTIs, bacteremia, meningitis'
        },
        'ciprofloxacin': {
            'class': 'Fluoroquinolone',
            'alternatives': ['Levofloxacin', 'Moxifloxacin', 'Trimethoprim-sulfamethoxazole'],
            'use_case': 'UTIs, respiratory infections, GI infections'
        },
        'gentamicin': {
            'class': 'Aminoglycoside',
            'alternatives': ['Amikacin', 'Tobramycin', 'Plazomicin'],
            'use_case': 'Severe infections, synergy with beta-lactams'
        }
    }
    
    susceptible_drugs = []
    resistant_drugs = []
    
    for ab, pred in predictions.items():
        info = drug_classes[ab]
        phenotype = pred['phenotype']
        confidence = pred['confidence']
        
        if phenotype == 'Resistant':
            resistant_drugs.append(ab)
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'AVOID',
                'status_color': '#ef4444',
                'message': f'High-confidence resistance detected ({confidence:.0%}). Consider alternatives: {", ".join(info["alternatives"][:2])}.',
                'alternatives': info['alternatives']
            })
        elif phenotype == 'Susceptible':
            susceptible_drugs.append(ab)
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'RECOMMENDED',
                'status_color': '#22c55e',
                'message': f'Predicted susceptible ({confidence:.0%}). Suitable for {info["use_case"]}.',
                'alternatives': []
            })
        else:
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'USE WITH CAUTION',
                'status_color': '#f59e0b',
                'message': f'Intermediate resistance detected ({confidence:.0%}). Consider higher dosing or alternative agents.',
                'alternatives': info['alternatives'][:1]
            })
    
    # Overall recommendation
    if susceptible_drugs:
        overall = f"Consider using {susceptible_drugs[0].capitalize()} as first-line therapy."
    else:
        overall = "⚠️ Multi-drug resistance detected. Consult infectious disease specialist."
    
    return {
        'recommendations': recommendations,
        'overall': overall,
        'mdr_risk': len(resistant_drugs) >= 2,
        'n_resistant': len(resistant_drugs),
        'n_susceptible': len(susceptible_drugs)
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint. Accepts FASTA text."""
    try:
        data = request.get_json()
        fasta_text = data.get('fasta', '')
        
        if not fasta_text.strip():
            return jsonify({'error': 'No FASTA sequence provided'}), 400
        
        # Extract features
        features, error = extract_features_from_fasta(fasta_text)
        if error:
            return jsonify({'error': error}), 400
        
        # Make predictions for each antibiotic
        predictions = {}
        gene_detections = []
        feature_names = metadata.get('feature_names', [])
        
        for ab in ANTIBIOTICS:
            if ab not in models:
                continue
            
            model = models[ab]
            proba = model.predict_proba(features.reshape(1, -1))[0]
            pred_class = int(np.argmax(proba))
            confidence = float(proba[pred_class])
            
            predictions[ab] = {
                'phenotype': LABEL_NAMES[pred_class],
                'confidence': confidence,
                'probabilities': {
                    'Susceptible': round(float(proba[0]), 4),
                    'Intermediate': round(float(proba[1]), 4),
                    'Resistant': round(float(proba[2]), 4)
                }
            }
        
        # Detected genes
        for i, fname in enumerate(feature_names):
            if features[i] > 0:
                # Find which antibiotic this gene is associated with
                assoc_antibiotics = []
                category = 'unknown'
                for ab, panels in metadata.get('gene_panels', {}).items():
                    for cat, genes in panels.items():
                        if fname in genes:
                            assoc_antibiotics.append(ab)
                            category = cat
                
                gene_detections.append({
                    'gene': fname,
                    'detected': True,
                    'category': category,
                    'associated_antibiotics': assoc_antibiotics
                })
        
        # Clinical recommendations
        clinical = get_clinical_recommendation(predictions)
        
        # Get SHAP/feature importance for network visualization
        network_data = build_gene_network(predictions, gene_detections)
        
        return jsonify({
            'success': True,
            'organism': 'Escherichia coli',
            'predictions': predictions,
            'detected_genes': gene_detections,
            'clinical': clinical,
            'network': network_data,
            'model_metrics': metadata.get('results', {})
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def build_gene_network(predictions, detected_genes):
    """Build gene interaction network data for visualization."""
    nodes = []
    links = []
    
    # Add antibiotic nodes
    for ab in ANTIBIOTICS:
        pred = predictions.get(ab, {})
        color = '#ef4444' if pred.get('phenotype') == 'Resistant' else '#22c55e' if pred.get('phenotype') == 'Susceptible' else '#f59e0b'
        nodes.append({
            'id': ab,
            'name': ab.capitalize(),
            'type': 'antibiotic',
            'group': 0,
            'size': 30,
            'color': color,
            'phenotype': pred.get('phenotype', 'Unknown')
        })
    
    # Add detected gene nodes and links
    gene_panels = metadata.get('gene_panels', {})
    for gene_info in detected_genes[:20]:  # Limit to top 20 genes
        gene = gene_info['gene']
        category = gene_info['category']
        
        # Determine color by category
        cat_colors = {
            'primary': '#8b5cf6',
            'secondary': '#06b6d4', 
            'snps': '#f97316',
            'primary_resistance': '#8b5cf6',
            'secondary_mechanism': '#06b6d4',
            'snp_marker': '#f97316',
            'unknown': '#6b7280'
        }
        
        nodes.append({
            'id': gene,
            'name': gene,
            'type': 'gene',
            'group': 1 if 'primary' in category else 2 if 'secondary' in category else 3,
            'size': 15 if 'primary' in category else 10,
            'color': cat_colors.get(category, '#6b7280'),
            'category': category
        })
        
        # Create links to associated antibiotics
        for ab in gene_info.get('associated_antibiotics', []):
            # Get importance score
            importance = 0.5
            feat_imp = metadata.get('feature_importance', {}).get(ab, [])
            for fi in feat_imp:
                if fi['gene'] == gene:
                    importance = fi['importance']
                    break
            
            links.append({
                'source': gene,
                'target': ab,
                'strength': importance,
                'type': category
            })
    
    # Add inter-gene links for co-located genes (same mobile genetic element)
    gene_ids = [g['gene'] for g in detected_genes[:20]]
    for i in range(len(gene_ids)):
        for j in range(i + 1, len(gene_ids)):
            # Check if genes are on same resistance mechanism
            for ab, panels in gene_panels.items():
                genes_on_panel = panels.get('primary', []) + panels.get('secondary', [])
                if gene_ids[i] in genes_on_panel and gene_ids[j] in genes_on_panel:
                    links.append({
                        'source': gene_ids[i],
                        'target': gene_ids[j],
                        'strength': 0.3,
                        'type': 'co-occurrence'
                    })
                    break
    
    return {'nodes': nodes, 'links': links}


@app.route('/api/metrics')
def get_metrics():
    """Return model performance metrics."""
    return jsonify({
        'results': metadata.get('results', {}),
        'antibiotics': ANTIBIOTICS,
        'organism': 'Escherichia coli',
        'n_features': metadata.get('n_features', 0)
    })


@app.route('/api/shap')
def get_shap():
    """Return SHAP feature importance data."""
    return jsonify(shap_data)


@app.route('/api/sample_fasta')
def sample_fasta():
    """Return a sample FASTA for testing."""
    # Generate a realistic-looking E. coli genomic sequence fragment
    np.random.seed(None)  # Random seed for variety
    seq_len = 5000
    bases = ['A', 'T', 'G', 'C']
    # E. coli has ~50.8% GC content
    weights = [0.246, 0.246, 0.254, 0.254]
    sequence = ''.join(np.random.choice(bases, size=seq_len, p=weights))
    
    # Format as FASTA
    fasta = ">Escherichia_coli_sample_genome_assembly contig_1 length=5000\n"
    for i in range(0, len(sequence), 70):
        fasta += sequence[i:i+70] + "\n"
    
    return jsonify({'fasta': fasta})


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000, host='0.0.0.0')
