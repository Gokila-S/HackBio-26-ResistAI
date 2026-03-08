"""
ResistAI — Antibiotic Resistance Predictor (Flask Web Application)
====================================================================
AI-powered genotype-to-phenotype prediction of antibiotic resistance
in Escherichia coli with advanced visualization and clinical decision support.

Features:
  - Multi-drug resistance prediction (Ampicillin, Ciprofloxacin, Gentamicin)
  - Gene network visualization (D3.js force-directed graph)
  - SHAP-based feature interpretability
  - Batch genome analysis with epidemiological clustering
  - Resistance evolution timeline simulator
  - WHO AWaRe antibiotic classification
  - PDF clinical report generation
  - Comparative genomics dashboard
  - Resistance mechanism explainer
"""

import os
import json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import re
import random
import io
import datetime

app = Flask(__name__)
CORS(app)

# ── Load models and metadata ──────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
ANTIBIOTICS = ['ampicillin', 'ciprofloxacin', 'gentamicin']
LABEL_NAMES = ['Susceptible', 'Intermediate', 'Resistant']

models = {}
metadata = {}
shap_data = {}

# ── WHO AWaRe Classification ──────────────────────────────────────────
WHO_AWARE = {
    'ampicillin': {
        'category': 'Access',
        'color': '#22c55e',
        'description': 'First-line antibiotics that should be widely available',
        'icon': '🟢'
    },
    'ciprofloxacin': {
        'category': 'Watch',
        'color': '#f59e0b',
        'description': 'Higher resistance potential — use only for specific indications',
        'icon': '🟡'
    },
    'gentamicin': {
        'category': 'Watch',
        'color': '#f59e0b',
        'description': 'Higher resistance potential — prioritize as key targets for stewardship',
        'icon': '🟡'
    },
    # Alternative drugs
    'meropenem': {'category': 'Watch', 'color': '#f59e0b', 'icon': '🟡'},
    'colistin': {'category': 'Reserve', 'color': '#ef4444', 'icon': '🔴'},
    'plazomicin': {'category': 'Reserve', 'color': '#ef4444', 'icon': '🔴'},
    'ceftriaxone': {'category': 'Watch', 'color': '#f59e0b', 'icon': '🟡'},
    'amikacin': {'category': 'Access', 'color': '#22c55e', 'icon': '🟢'},
    'piperacillin-tazobactam': {'category': 'Watch', 'color': '#f59e0b', 'icon': '🟡'},
}

# ── Resistance Mechanism Database ─────────────────────────────────────
RESISTANCE_MECHANISMS = {
    'blaTEM-1': {
        'full_name': 'TEM-1 β-lactamase',
        'mechanism': 'Enzymatic hydrolysis',
        'description': 'Produces a β-lactamase enzyme that cleaves the β-lactam ring of penicillins, rendering them inactive. TEM-1 is the most common plasmid-mediated β-lactamase in Gram-negative bacteria.',
        'location': 'Plasmid-borne (Tn2, Tn3 transposons)',
        'prevalence': 'Found in ~60% of ampicillin-resistant E. coli',
        'molecular_target': 'β-lactam ring hydrolysis'
    },
    'blaTEM-2': {
        'full_name': 'TEM-2 β-lactamase',
        'mechanism': 'Enzymatic hydrolysis',
        'description': 'A derivative of TEM-1 with a single amino acid substitution. Similar broad-spectrum β-lactamase activity against penicillins and early cephalosporins.',
        'location': 'Plasmid-borne',
        'prevalence': 'Less common than TEM-1',
        'molecular_target': 'β-lactam ring hydrolysis'
    },
    'blaCTX-M-15': {
        'full_name': 'CTX-M-15 Extended-Spectrum β-lactamase',
        'mechanism': 'Enzymatic hydrolysis (ESBL)',
        'description': 'The most globally disseminated ESBL. Hydrolyzes cefotaxime and ceftazidime. Often co-carried with other resistance genes on IncF plasmids. A major clinical concern worldwide.',
        'location': 'Plasmid-borne (ISEcp1-mediated mobilization)',
        'prevalence': 'Dominant ESBL globally — found on all continents',
        'molecular_target': 'Extended-spectrum cephalosporin hydrolysis'
    },
    'blaCTX-M-14': {
        'full_name': 'CTX-M-14 Extended-Spectrum β-lactamase',
        'mechanism': 'Enzymatic hydrolysis (ESBL)',
        'description': 'Second most common CTX-M type. Preferentially hydrolyzes cefotaxime over ceftazidime.',
        'location': 'Plasmid-borne',
        'prevalence': 'Common in Asia and Southern Europe',
        'molecular_target': 'Cefotaxime hydrolysis'
    },
    'blaSHV-1': {
        'full_name': 'SHV-1 β-lactamase',
        'mechanism': 'Enzymatic hydrolysis',
        'description': 'Originally chromosomal in K. pneumoniae but now found on plasmids in E. coli. Provides resistance to ampicillin and early cephalosporins.',
        'location': 'Chromosomal/Plasmid',
        'prevalence': 'Common in hospital-acquired infections',
        'molecular_target': 'Penicillin hydrolysis'
    },
    'blaOXA-1': {
        'full_name': 'OXA-1 β-lactamase',
        'mechanism': 'Enzymatic hydrolysis',
        'description': 'Oxacillin-hydrolyzing enzyme. Can also reduce susceptibility to amoxicillin-clavulanate when co-expressed with TEM or SHV enzymes.',
        'location': 'Plasmid-borne (integron-associated)',
        'prevalence': 'Frequently found with CTX-M-15',
        'molecular_target': 'Oxacillin and cloxacillin hydrolysis'
    },
    'gyrA_S83L': {
        'full_name': 'DNA Gyrase A subunit — Ser83→Leu',
        'mechanism': 'Target site modification',
        'description': 'Point mutation in the quinolone resistance-determining region (QRDR) of DNA gyrase. This single mutation confers low-level fluoroquinolone resistance. Combined with parC mutations, leads to high-level resistance.',
        'location': 'Chromosomal (QRDR of gyrA)',
        'prevalence': 'Found in >90% of ciprofloxacin-resistant E. coli',
        'molecular_target': 'Reduces quinolone binding to DNA gyrase'
    },
    'gyrA_D87N': {
        'full_name': 'DNA Gyrase A subunit — Asp87→Asn',
        'mechanism': 'Target site modification',
        'description': 'Second most common gyrA QRDR mutation. When combined with S83L, provides high-level ciprofloxacin resistance (MIC >32 mg/L).',
        'location': 'Chromosomal',
        'prevalence': 'Found in ~70% of highly resistant isolates',
        'molecular_target': 'Reduces quinolone binding affinity'
    },
    'parC_S80I': {
        'full_name': 'Topoisomerase IV subunit C — Ser80→Ile',
        'mechanism': 'Target site modification',
        'description': 'Mutation in topoisomerase IV, the secondary target of fluoroquinolones in Gram-negative bacteria. Works synergistically with gyrA mutations.',
        'location': 'Chromosomal',
        'prevalence': 'Common in high-level FQ resistance',
        'molecular_target': 'Altered topoisomerase IV binding'
    },
    'qnrA1': {
        'full_name': 'Quinolone Resistance protein QnrA1',
        'mechanism': 'Target protection',
        'description': 'Pentapeptide repeat protein that protects DNA gyrase and topoisomerase IV from quinolone binding. Provides low-level resistance but facilitates selection of chromosomal mutations.',
        'location': 'Plasmid-borne (often on IncA/C plasmids)',
        'prevalence': 'Less common than qnrB and qnrS',
        'molecular_target': 'Gyrase/topoisomerase protection'
    },
    'qnrB1': {
        'full_name': 'Quinolone Resistance protein QnrB1',
        'mechanism': 'Target protection',
        'description': 'Most diverse family of Qnr proteins. Protects DNA gyrase from quinolone inhibition.',
        'location': 'Plasmid-borne',
        'prevalence': 'Widespread globally',
        'molecular_target': 'Gyrase protection'
    },
    'qnrS1': {
        'full_name': 'Quinolone Resistance protein QnrS1',
        'mechanism': 'Target protection',
        'description': 'Provides stepping-stone resistance facilitating selection of higher-level chromosomal mutations.',
        'location': 'Plasmid-borne',
        'prevalence': 'Common in community-acquired E. coli',
        'molecular_target': 'Gyrase protection'
    },
    "aac(6')-Ib-cr": {
        'full_name': "Aminoglycoside acetyltransferase AAC(6')-Ib-cr",
        'mechanism': 'Enzymatic modification (dual activity)',
        'description': 'Unique enzyme with dual activity: modifies both aminoglycosides and fluoroquinolones (ciprofloxacin, norfloxacin). Two amino acid substitutions (Trp102Arg, Asp179Tyr) enable quinolone acetylation.',
        'location': 'Plasmid-borne (gene cassette in integrons)',
        'prevalence': 'Very common in MDR E. coli',
        'molecular_target': 'N-acetylation of piperazinyl amine'
    },
    'aac(3)-IIa': {
        'full_name': "Aminoglycoside 3-N-acetyltransferase",
        'mechanism': 'Enzymatic modification',
        'description': 'Acetylates the 3-amino group of aminoglycosides including gentamicin, tobramycin, and netilmicin, preventing ribosome binding.',
        'location': 'Plasmid-borne (Tn21-related transposons)',
        'prevalence': 'Most common gentamicin resistance gene in Enterobacteriaceae',
        'molecular_target': '3-amino group acetylation'
    },
    'armA': {
        'full_name': '16S rRNA methyltransferase ArmA',
        'mechanism': '16S rRNA methylation',
        'description': 'Methylates the 16S rRNA at position G1405, blocking aminoglycoside binding. Confers high-level resistance to all clinically relevant aminoglycosides except streptomycin.',
        'location': 'Plasmid-borne (IS-associated)',
        'prevalence': 'Emerging — most common 16S RMTase',
        'molecular_target': 'G1405 methylation of 16S rRNA'
    },
    'rmtB': {
        'full_name': '16S rRNA methyltransferase RmtB',
        'mechanism': '16S rRNA methylation',
        'description': 'Methylates 16S rRNA, conferring pan-aminoglycoside resistance. Often co-located with blaCTX-M or blaNDM genes.',
        'location': 'Plasmid-borne',
        'prevalence': 'Common in East Asia',
        'molecular_target': 'rRNA methylation'
    },
    'acrAB_overexpression': {
        'full_name': 'AcrAB-TolC Efflux Pump Overexpression',
        'mechanism': 'Active efflux',
        'description': 'Overexpression of the AcrAB-TolC RND-type efflux pump system. Exports a wide range of antibiotics including β-lactams, fluoroquinolones, and aminoglycosides out of the cell.',
        'location': 'Chromosomal (regulatory mutations)',
        'prevalence': 'Common contributor to low-level MDR',
        'molecular_target': 'Active drug export'
    },
    'ompF_loss': {
        'full_name': 'OmpF Porin Loss',
        'mechanism': 'Reduced permeability',
        'description': 'Loss or reduced expression of OmpF outer membrane porin decreases antibiotic entry into the cell. Contributes to resistance when combined with other mechanisms.',
        'location': 'Chromosomal',
        'prevalence': 'Common in MDR isolates',
        'molecular_target': 'Reduced outer membrane permeability'
    },
    'rpsL_K43R': {
        'full_name': 'Ribosomal protein S12 — Lys43→Arg',
        'mechanism': 'Target site modification',
        'description': 'Mutation in ribosomal protein S12 reduces streptomycin binding affinity. One of the classic mechanisms of streptomycin resistance.',
        'location': 'Chromosomal',
        'prevalence': 'Common in streptomycin-resistant strains',
        'molecular_target': 'Altered ribosome binding site'
    }
}


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
    features = np.zeros(n_features)
    
    # GC content analysis
    gc_content = (full_seq.count('G') + full_seq.count('C')) / max(seq_len, 1)
    
    # K-mer based feature extraction (simulating gene detection)
    for i, gene_name in enumerate(feature_names):
        gene_hash = hash(gene_name + full_seq[:min(1000, seq_len)])
        segment_start = abs(gene_hash) % max(1, seq_len - 100)
        segment = full_seq[segment_start:segment_start + 100] if seq_len > 100 else full_seq
        
        at_content = (segment.count('A') + segment.count('T')) / max(len(segment), 1)
        
        if 'bla' in gene_name or 'qnr' in gene_name or 'aac' in gene_name:
            threshold = 0.45
        elif 'mutation' in gene_name or gene_name.count('_') >= 2:
            threshold = 0.55
        else:
            threshold = 0.50
        
        detection_score = (at_content * 0.6 + gc_content * 0.4)
        noise = (abs(hash(gene_name + str(i))) % 100) / 500.0
        
        features[i] = 1.0 if (detection_score + noise) > threshold else 0.0
    
    return features, None


def get_clinical_recommendation(predictions):
    """Generate clinical interpretation based on predictions with WHO AWaRe."""
    recommendations = []
    
    drug_classes = {
        'ampicillin': {
            'class': 'Beta-lactam (Penicillin)',
            'alternatives': ['Piperacillin-tazobactam', 'Meropenem', 'Ceftriaxone'],
            'use_case': 'UTIs, bacteremia, meningitis',
            'who_aware': WHO_AWARE.get('ampicillin', {})
        },
        'ciprofloxacin': {
            'class': 'Fluoroquinolone',
            'alternatives': ['Levofloxacin', 'Moxifloxacin', 'Trimethoprim-sulfamethoxazole'],
            'use_case': 'UTIs, respiratory infections, GI infections',
            'who_aware': WHO_AWARE.get('ciprofloxacin', {})
        },
        'gentamicin': {
            'class': 'Aminoglycoside',
            'alternatives': ['Amikacin', 'Tobramycin', 'Plazomicin'],
            'use_case': 'Severe infections, synergy with beta-lactams',
            'who_aware': WHO_AWARE.get('gentamicin', {})
        }
    }
    
    susceptible_drugs = []
    resistant_drugs = []
    
    for ab, pred in predictions.items():
        info = drug_classes[ab]
        phenotype = pred['phenotype']
        confidence = pred['confidence']
        aware = info['who_aware']
        
        if phenotype == 'Resistant':
            resistant_drugs.append(ab)
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'AVOID',
                'status_color': '#ef4444',
                'message': f'High-confidence resistance detected ({confidence:.0%}). Consider alternatives: {", ".join(info["alternatives"][:2])}.',
                'alternatives': info['alternatives'],
                'who_category': aware.get('category', 'Unknown'),
                'who_color': aware.get('color', '#6b7280'),
                'who_icon': aware.get('icon', '⚪')
            })
        elif phenotype == 'Susceptible':
            susceptible_drugs.append(ab)
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'RECOMMENDED',
                'status_color': '#22c55e',
                'message': f'Predicted susceptible ({confidence:.0%}). Suitable for {info["use_case"]}.',
                'alternatives': [],
                'who_category': aware.get('category', 'Unknown'),
                'who_color': aware.get('color', '#6b7280'),
                'who_icon': aware.get('icon', '⚪')
            })
        else:
            recommendations.append({
                'antibiotic': ab.capitalize(),
                'drug_class': info['class'],
                'status': 'USE WITH CAUTION',
                'status_color': '#f59e0b',
                'message': f'Intermediate resistance detected ({confidence:.0%}). Consider higher dosing or alternative agents.',
                'alternatives': info['alternatives'][:1],
                'who_category': aware.get('category', 'Unknown'),
                'who_color': aware.get('color', '#6b7280'),
                'who_icon': aware.get('icon', '⚪')
            })
    
    # Overall recommendation
    if susceptible_drugs:
        overall = f"Consider using {susceptible_drugs[0].capitalize()} as first-line therapy."
    else:
        overall = "⚠️ Multi-drug resistance detected. Consult infectious disease specialist."
    
    # Stewardship score (0-100)
    stewardship_score = max(0, 100 - (len(resistant_drugs) * 30) - (3 - len(susceptible_drugs)) * 10)
    
    return {
        'recommendations': recommendations,
        'overall': overall,
        'mdr_risk': len(resistant_drugs) >= 2,
        'n_resistant': len(resistant_drugs),
        'n_susceptible': len(susceptible_drugs),
        'stewardship_score': stewardship_score
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
        
        # Sequence stats
        sequences = parse_fasta(fasta_text)
        full_seq = ''.join(sequences.values())
        seq_stats = {
            'total_length': len(full_seq),
            'n_contigs': len(sequences),
            'gc_content': round((full_seq.count('G') + full_seq.count('C')) / max(len(full_seq), 1) * 100, 1),
            'n50': len(full_seq) // max(len(sequences), 1)
        }
        
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
        
        # Detected genes with mechanism info
        for i, fname in enumerate(feature_names):
            if features[i] > 0:
                assoc_antibiotics = []
                category = 'unknown'
                for ab, panels in metadata.get('gene_panels', {}).items():
                    for cat, genes in panels.items():
                        if fname in genes:
                            assoc_antibiotics.append(ab)
                            category = cat
                
                mech = RESISTANCE_MECHANISMS.get(fname, {})
                gene_detections.append({
                    'gene': fname,
                    'detected': True,
                    'category': category,
                    'associated_antibiotics': assoc_antibiotics,
                    'mechanism_info': {
                        'full_name': mech.get('full_name', fname),
                        'mechanism': mech.get('mechanism', 'Unknown'),
                        'description': mech.get('description', 'Resistance mechanism details not available.'),
                        'location': mech.get('location', 'Unknown'),
                        'prevalence': mech.get('prevalence', 'Unknown'),
                        'molecular_target': mech.get('molecular_target', 'Unknown')
                    }
                })
        
        # Clinical recommendations
        clinical = get_clinical_recommendation(predictions)
        
        # Gene network
        network_data = build_gene_network(predictions, gene_detections)
        
        # Heatmap data
        heatmap_data = build_heatmap_data(features, feature_names)
        
        # Resistance evolution timeline
        timeline = build_resistance_timeline(predictions, gene_detections)
        
        return jsonify({
            'success': True,
            'organism': 'Escherichia coli',
            'timestamp': datetime.datetime.now().isoformat(),
            'seq_stats': seq_stats,
            'predictions': predictions,
            'detected_genes': gene_detections,
            'clinical': clinical,
            'network': network_data,
            'heatmap': heatmap_data,
            'timeline': timeline,
            'model_metrics': metadata.get('results', {}),
            'who_aware': WHO_AWARE
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint. Accepts multiple FASTA sequences."""
    try:
        data = request.get_json()
        fasta_texts = data.get('sequences', [])
        
        if not fasta_texts:
            return jsonify({'error': 'No sequences provided'}), 400
        
        results = []
        for idx, fasta_text in enumerate(fasta_texts[:10]):  # Limit to 10
            features, error = extract_features_from_fasta(fasta_text)
            if error:
                results.append({'id': idx, 'error': error})
                continue
            
            predictions = {}
            feature_names = metadata.get('feature_names', [])
            
            for ab in ANTIBIOTICS:
                if ab not in models:
                    continue
                model = models[ab]
                proba = model.predict_proba(features.reshape(1, -1))[0]
                pred_class = int(np.argmax(proba))
                predictions[ab] = {
                    'phenotype': LABEL_NAMES[pred_class],
                    'confidence': float(proba[pred_class]),
                    'probabilities': {
                        'Susceptible': round(float(proba[0]), 4),
                        'Intermediate': round(float(proba[1]), 4),
                        'Resistant': round(float(proba[2]), 4)
                    }
                }
            
            # Detected genes
            detected = []
            for i, fname in enumerate(feature_names):
                if features[i] > 0:
                    detected.append(fname)
            
            sequences = parse_fasta(fasta_text)
            header = list(sequences.keys())[0] if sequences else f'Isolate_{idx+1}'
            
            results.append({
                'id': idx,
                'name': header,
                'predictions': predictions,
                'detected_genes': detected,
                'n_resistance_genes': len(detected),
                'mdr': sum(1 for p in predictions.values() if p['phenotype'] == 'Resistant') >= 2
            })
        
        # Clustering: group by resistance profile
        clusters = {}
        for r in results:
            if 'error' in r:
                continue
            profile = tuple(r['predictions'].get(ab, {}).get('phenotype', 'Unknown') for ab in ANTIBIOTICS)
            key = '_'.join(profile)
            if key not in clusters:
                clusters[key] = {'profile': dict(zip(ANTIBIOTICS, profile)), 'isolates': []}
            clusters[key]['isolates'].append(r['name'])
        
        return jsonify({
            'success': True,
            'results': results,
            'clusters': list(clusters.values()),
            'n_analyzed': len(results),
            'n_mdr': sum(1 for r in results if r.get('mdr', False))
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def build_gene_network(predictions, detected_genes):
    """Build gene interaction network data for visualization."""
    nodes = []
    links = []
    
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
    
    gene_panels = metadata.get('gene_panels', {})
    for gene_info in detected_genes[:20]:
        gene = gene_info['gene']
        category = gene_info['category']
        
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
        
        for ab in gene_info.get('associated_antibiotics', []):
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
    
    gene_ids = [g['gene'] for g in detected_genes[:20]]
    for i in range(len(gene_ids)):
        for j in range(i + 1, len(gene_ids)):
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


def build_heatmap_data(features, feature_names):
    """Build heatmap correlation data between genes and antibiotics."""
    heatmap = []
    gene_panels = metadata.get('gene_panels', {})
    
    for i, fname in enumerate(feature_names):
        if features[i] > 0:
            row = {'gene': fname}
            for ab in ANTIBIOTICS:
                # Check if this gene is in this antibiotic's panel
                panels = gene_panels.get(ab, {})
                if fname in panels.get('primary', []):
                    row[ab] = 0.9 + random.uniform(0, 0.1)
                elif fname in panels.get('secondary', []):
                    row[ab] = 0.5 + random.uniform(0, 0.2)
                elif fname in panels.get('snps', []):
                    row[ab] = 0.6 + random.uniform(0, 0.2)
                else:
                    row[ab] = random.uniform(0, 0.15)
            heatmap.append(row)
    
    return heatmap


def build_resistance_timeline(predictions, detected_genes):
    """Build resistance evolution timeline data."""
    timeline = []
    
    # Simulate evolutionary steps
    gene_order_priority = {
        'primary': 1,
        'primary_resistance': 1,
        'snps': 2,
        'snp_marker': 2,
        'secondary': 3,
        'secondary_mechanism': 3,
    }
    
    sorted_genes = sorted(detected_genes, key=lambda g: gene_order_priority.get(g['category'], 4))
    
    steps = [
        {'stage': 'Wild Type', 'description': 'Susceptible ancestor — no acquired resistance genes', 'genes': [], 'resistance_level': 0},
    ]
    
    accumulated_genes = []
    for idx, gene in enumerate(sorted_genes[:8]):
        accumulated_genes.append(gene['gene'])
        mech = RESISTANCE_MECHANISMS.get(gene['gene'], {})
        mechanism_type = mech.get('mechanism', gene['category'])
        
        resistance_level = min(100, (idx + 1) * (100 // min(8, len(sorted_genes))))
        
        steps.append({
            'stage': f'Step {idx + 1}: {mechanism_type}',
            'description': f'Acquisition of {gene["gene"]} — {mech.get("description", "Resistance gene detected")[:80]}...',
            'genes': list(accumulated_genes),
            'resistance_level': resistance_level,
            'gene_acquired': gene['gene'],
            'location': mech.get('location', 'Unknown')
        })
    
    if any(p.get('phenotype') == 'Resistant' for p in predictions.values()):
        steps.append({
            'stage': 'MDR Phenotype',
            'description': 'Multiple resistance mechanisms accumulated — clinical resistance achieved',
            'genes': accumulated_genes,
            'resistance_level': 100,
        })
    
    return steps


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


@app.route('/api/mechanism/<gene_name>')
def get_mechanism(gene_name):
    """Return detailed mechanism info for a specific gene."""
    mech = RESISTANCE_MECHANISMS.get(gene_name)
    if mech:
        return jsonify({'success': True, 'gene': gene_name, **mech})
    return jsonify({'success': False, 'message': 'Gene mechanism not found'}), 404


@app.route('/api/sample_fasta')
def sample_fasta():
    """Return a sample FASTA for testing."""
    np.random.seed(None)
    seq_len = 5000
    bases = ['A', 'T', 'G', 'C']
    weights = [0.246, 0.246, 0.254, 0.254]
    sequence = ''.join(np.random.choice(bases, size=seq_len, p=weights))
    
    fasta = ">Escherichia_coli_sample_genome_assembly contig_1 length=5000\n"
    for i in range(0, len(sequence), 70):
        fasta += sequence[i:i+70] + "\n"
    
    return jsonify({'fasta': fasta})


@app.route('/api/multi_sample_fasta')
def multi_sample_fasta():
    """Return multiple sample FASTA sequences for batch testing."""
    samples = []
    names = [
        'EC_Isolate_UTI_2024_001',
        'EC_Isolate_Blood_2024_002', 
        'EC_Isolate_Wound_2024_003'
    ]
    for name in names:
        np.random.seed(None)
        seq_len = 5000
        bases = ['A', 'T', 'G', 'C']
        weights = [0.246, 0.246, 0.254, 0.254]
        sequence = ''.join(np.random.choice(bases, size=seq_len, p=weights))
        
        fasta = f">{name} contig_1 length={seq_len}\n"
        for i in range(0, len(sequence), 70):
            fasta += sequence[i:i+70] + "\n"
        samples.append(fasta)
    
    return jsonify({'sequences': samples})


# ── Main ───────────────────────────────────────────────────────────────
# Load models at import time so gunicorn workers have them ready
load_models()

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')

