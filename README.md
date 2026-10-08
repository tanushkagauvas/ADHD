# ADHD EEG Functional Connectivity & Graph Neural Network (GCN)

This repository contains the preprocessing, functional connectivity analysis, and graph construction pipeline for classification of Attention Deficit Hyperactivity Disorder (ADHD) using multichannel EEG data.

---

## 📌 Project Architecture

```
ADHD/
├── data/
│   ├── raw/             # Raw EEG datasets (ignored in git)
│   └── processed/       # Extracted epochs and connectivity matrices (ignored in git)
├── models/              # GCN model architectures and checkpoints
├── results/             # Statistical summaries and visualization plots
├── src/
│   └── person1/         # Pipeline modules
│       ├── load_and_segment.py   # EEG loading, epoching, and filtering
│       ├── coherence.py          # Spectral Coherence computation
│       ├── pli.py                # Phase Lag Index (PLI) computation
│       ├── graph_builder.py      # Adjacency matrices & PyG graph construction
│       └── visualize.py          # Topographic maps & connectivity heatmaps
├── tests/
│   └── generate_mock_data.py    # Synthetic EEG generator for testing
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/tanushkagauvas/<repo-name>.git
cd ADHD
pip install -r requirements.txt
```

### 2. Dataset Setup

Place your raw EEG dataset (`adhdata.csv`) into the `data/raw/` directory:

```
data/
└── raw/
    └── adhdata.csv
```

> **Note**: Raw data (`*.csv`) and processed tensors (`*.npy`) are excluded from this repository due to file size limits and data privacy.

### 3. Running the Pipeline

1. **Generate Mock Data (Optional for testing):**
   ```bash
   python tests/generate_mock_data.py
   ```

2. **Process EEG & Extract Connectivity:**
   ```bash
   python src/person1/load_and_segment.py
   python src/person1/coherence.py
   python src/person1/pli.py
   python src/person1/graph_builder.py
   ```

3. **Generate Visualizations:**
   ```bash
   python src/person1/visualize.py
   ```
