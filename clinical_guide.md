# Clinical Interpretation Guide — ResistAI

## Purpose

This guide helps clinicians interpret the output of ResistAI's antibiotic resistance predictions to support informed antibiotic selection for *Escherichia coli* infections.

---

## How to Use ResistAI

### Step 1: Input
Provide a bacterial genome assembly in **FASTA format**. This can be obtained from whole-genome sequencing (WGS) platforms such as Illumina MiSeq/NextSeq or Oxford Nanopore.

### Step 2: Interpretation
ResistAI analyzes 51 genomic features including:
- **Primary resistance genes** (e.g., blaTEM, blaCTX-M, qnr genes)
- **Secondary resistance mechanisms** (efflux pumps, porin loss)
- **SNP markers** (target site mutations in gyrA, parC, rpsL)

### Step 3: Output
For each antibiotic, the tool provides:
- **Phenotype prediction**: Resistant (R), Susceptible (S), or Intermediate (I)
- **Confidence score**: Model certainty (0-100%)
- **Probability distribution**: Likelihood of each phenotype class

---

## Antibiotic-Specific Guidance

### Ampicillin (Beta-lactam / Penicillin)

| Prediction | Clinical Action |
|------------|----------------|
| **Susceptible** | Safe to use for uncomplicated UTIs, intra-abdominal infections |
| **Intermediate** | Consider higher dosing (2g q4h IV) or switch to amoxicillin-clavulanate |
| **Resistant** | AVOID. Consider: Piperacillin-tazobactam, Ceftriaxone, or Meropenem |

**Key resistance genes detected:**
- `blaTEM-1/2` — TEM beta-lactamase (most common)
- `blaCTX-M-15/14` — Extended-spectrum beta-lactamase (ESBL)
- `blaSHV-1` — SHV beta-lactamase
- `blaOXA-1` — OXA-type beta-lactamase
- `blaCMY-2` — AmpC-type cephalosporinase

### Ciprofloxacin (Fluoroquinolone)

| Prediction | Clinical Action |
|------------|----------------|
| **Susceptible** | Effective for UTIs, traveler's diarrhea, respiratory infections |
| **Intermediate** | Use with caution. Consider levofloxacin as alternative |
| **Resistant** | AVOID. Consider: Trimethoprim-sulfamethoxazole, Nitrofurantoin (for UTIs) |

**Key resistance markers:**
- `gyrA_S83L`, `gyrA_D87N` — DNA gyrase mutations (high-level resistance)
- `parC_S80I`, `parC_E84V` — Topoisomerase IV mutations
- `qnrA1/B1/S1` — Plasmid-mediated quinolone resistance
- `aac(6')-Ib-cr` — Aminoglycoside acetyltransferase (modifies ciprofloxacin)

### Gentamicin (Aminoglycoside)

| Prediction | Clinical Action |
|------------|----------------|
| **Susceptible** | Useful for severe infections, often in combination with beta-lactams |
| **Intermediate** | Monitor levels closely, consider extending interval dosing |
| **Resistant** | AVOID. Consider: Amikacin, Tobramycin, or Plazomicin |

**Key resistance markers:**
- `aac(3)-IIa/IVa` — Aminoglycoside acetyltransferases
- `ant(2'')-Ia` — Aminoglycoside nucleotidyltransferase
- `armA`, `rmtB` — 16S rRNA methyltransferases (high-level resistance)
- `aph(3')-Ia` — Aminoglycoside phosphotransferase

---

## Multi-Drug Resistance (MDR) Alert

If resistance is predicted for **2 or more antibiotics**, the system flags a **multi-drug resistance (MDR) warning**. In this case:

1. **Consult infectious disease specialist**
2. Consider carbapenem therapy (Meropenem, Imipenem)
3. Verify with culture-based AST when available
4. Report to institutional antibiogram committee

---

## Important Caveats

> **This tool is for RESEARCH and DECISION SUPPORT only.**

1. **Not a replacement for culture-based testing** — Always confirm with phenotypic antimicrobial susceptibility testing (AST)
2. **Confidence threshold** — Predictions with <80% confidence should be treated with caution
3. **Novel resistance mechanisms** — The model may not detect resistance mediated by previously uncharacterized genes
4. **Species specificity** — This model is trained on *E. coli* only. Do not use for other species
5. **Genomic data quality** — Accuracy depends on genome assembly quality and completeness

---

## Interpretation of Confidence Scores

| Confidence | Interpretation |
|------------|----------------|
| >90%       | High confidence — prediction is reliable |
| 70-90%     | Moderate confidence — consider clinical context |
| 50-70%     | Low confidence — recommend phenotypic confirmation |
| <50%       | Uncertain — do not base treatment decisions on this alone |

---

## Model Validation Metrics

The models were validated using 5-fold cross-validation on a dataset of 6,000 *E. coli* isolates with phenotypic resistance data from the BV-BRC (PATRIC) database.

| Metric | Ampicillin | Ciprofloxacin | Gentamicin |
|--------|-----------|---------------|------------|
| AUC    | 0.989     | 0.986         | 0.988      |
| MCC    | 0.862     | 0.874         | 0.906      |
| Accuracy | 91.5%   | 92.3%         | 94.3%      |

---

## Contact

For questions about clinical interpretation, contact the development team or refer to:
- [CLSI Breakpoint Tables](https://clsi.org/)
- [EUCAST Clinical Breakpoints](https://www.eucast.org/)
- [CARD Database](https://card.mcmaster.ca/) for AMR gene reference
