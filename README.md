# SubFraudGMM

A semi-supervised fraud detection approach in Brazilian public procurement using Gaussian Mixture Models and feature-subset ensembling.

---

## Method Overview

SubFraudGMM detects fraudulent procurement bids without labelled training data.
The algorithm proceeds in five steps:

1. **Feature subset generation** — Enumerate all non-empty subsets of 8 procurement features (255 subsets per dataset).
2. **BIC-optimal GMM** — For each subset, fit a Gaussian Mixture Model selecting the number of components and covariance type that minimises BIC (grid: 1–6 components × 4 covariance types).
3. **Threshold filtering** — Retain only subsets where at least one GMM cluster concentrates a known fraud proportion above a threshold *τ* ∈ {50, 60, 70, 80, 90, 100} %.
4. **LOO validation** — For each retained subset, apply Leave-One-Out cross-validation over confirmed fraud records to measure cluster consistency.
5. **Risk Indicator aggregation** — Combine three normalised signals across all subsets:
   - *Normalised Occurrence count* (how many subsets flagged the record)
   - *Normalised LOO score* (average cluster consistency)
   - *1 − Normalised Euclidean distance* to the fraud centroid

---

## Dataset

**Operação Patrola** — Brazilian Federal Police investigation into collusive bidding
in the rental of heavy road-construction machinery by municipal governments.

| Product | Records | Fraud records |
|---|---|---|
| Hydraulic excavator (*escavadeira*) | 4 651 | 33 |
| Motor grader (*motoniveladora*) | 403 | 22 |
| Compaction roller (*rolo compactador*) | 911 | 19 |
| Track-type tractor (*trator de esteira*) | 286 | 18 |

Source: eSfinge procurement system (Tribunal de Contas do Estado de São Paulo).

---

## Features

| Feature | Description |
|---|---|
| `unit_price` | Unit price quoted (R$) |
| `num_partic` | Number of bidders in the procurement process |
| `win` | Proportion of winning bids by the supplier at the same managing unit |
| `met` | Binary: supplier has previously contracted with this managing unit |
| `num` | Number of procurement processes the supplier has participated in |
| `period` | Supplier's active lifespan in the dataset (days) |
| `duration` | Duration of the procurement process (days from opening to homologation) |
| `unique` | Proportion of unique winners among all procurements at the managing unit |

---

## Benchmark

SubFraudGMM is compared against five deep anomaly detection models via
**ADBench** (Han et al., NeurIPS 2022) and a simple weighted-feature ranking baseline.

> Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022).
> ADBench: Anomaly Detection Benchmark.
> *Advances in Neural Information Processing Systems*, 35.
> https://github.com/Minqi824/ADBench

### AUC-ROC Summary (k = 100)

| Model | Escavadeira | Motoniveladora | Rolo compactador | Trator esteira |
|---|---|---|---|---|
| **SubFraudGMM** | **0.994** | **0.963** | **0.981** | **0.976** |
| REPEN | 0.976 | 0.831 | 0.938 | 0.887 |
| XGBOD | 0.989 | 0.935 | 0.972 | 0.921 |
| DevNet | 0.963 | 0.942 | 0.972 | 0.934 |
| FEAWAD | 0.949 | 0.597 | 0.824 | 0.854 |
| PReNet | 0.172 | 0.914 | 0.946 | 0.518 |
| RankingSimples | 0.810 | 0.332 | 0.792 | 0.527 |
| DeepSAD | 0.746 | 0.109 | 0.141 | 0.077 |

Full metrics table (AUC-ROC, AUC-PR, Precision@k, Recall@k, F1@k) in [`results/metrics_df.csv`](results/metrics_df.csv).

---

## Repository Structure

```
SubFraudGMM/
├── subfrauda_gmm.py              # SubFraudGMM algorithm as a Python module
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # PySpark preprocessing (requires HPC + eSfinge data)
│   ├── 02_model_training.ipynb       # Run SubFraudGMM across all datasets and thresholds
│   ├── 03_analysis.ipynb             # Load intermediary results, compute Risk Indicator
│   └── 04_benchmark_comparison.ipynb # Compare SubFraudGMM vs ADBench baselines
├── data/                         # Pre-processed dataset CSVs (4 files)
├── results/                      # Pre-computed model outputs and metrics
│   ├── SubFraudGMM.csv
│   ├── RankingSimples.csv
│   ├── DeepSAD.csv / DevNet.csv / FEAWAD.csv / PReNet.csv / REPEN.csv / XGBOD.csv
│   └── metrics_df.csv
├── requirements.txt
└── .gitignore
```

---

## How to Run

### Quick start — use pre-computed results

```bash
pip install -r requirements.txt
jupyter notebook notebooks/04_benchmark_comparison.ipynb
```

Cells 0–2 load the pre-computed CSVs from `results/` and require no additional data.

### Re-run SubFraudGMM from intermediate files

```bash
# 1. Run the analysis notebook (reads from results/intermediary/)
jupyter notebook notebooks/03_analysis.ipynb
```

### Re-run model training from scratch

```bash
# 2. Generate intermediate results (~5 hours on a 32-core machine)
jupyter notebook notebooks/02_model_training.ipynb
```

Outputs are written to `results/intermediary/` (~576 CSV files, excluded from git).

### Re-run data preprocessing (HPC only)

Notebook 01 requires PySpark and the raw eSfinge data files available only in the
original HPC cluster environment. The preprocessed CSVs in `data/` are provided
so that notebooks 02–04 can be run standalone.

---

## Results

Full benchmark metrics at `k = 100` (top-100 ranked records evaluated):

| Model | Produto | AUC-ROC | AUC-PR | Precision@k | Recall@k | F1@k |
|---|---|---|---|---|---|---|
| SubFraudGMM | Escavadeira | 0.994 | 0.032 | 0.02 | 0.061 | 0.030 |
| SubFraudGMM | Motoniveladora | 0.963 | 0.059 | 0.10 | 0.455 | 0.164 |
| SubFraudGMM | Rolo compactador | 0.981 | 0.076 | 0.10 | 0.526 | 0.168 |
| SubFraudGMM | Trator esteira | 0.976 | 0.082 | 0.13 | 0.722 | 0.220 |

SubFraudGMM achieves the highest AUC-ROC across all four product categories and
the highest Recall@100 for three of four categories, demonstrating strong
unsupervised fraud detection without requiring labelled training data.

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{subfrauda_gmm,
  author  = {Fernando {[Last Name]}},
  title   = {SubFraudGMM: Unsupervised Fraud Detection in Brazilian Public Procurement
             via Gaussian Mixture Model Ensembling},
  school  = {[University Name]},
  year    = {2025},
}
```
