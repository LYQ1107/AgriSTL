<div align="center">

# AgriSTL
### Agricultural Spatio-Temporal Learning for Plant Growth Prediction and Yield Estimation

<p>
  <img src="https://img.shields.io/badge/task-plant%20growth%20prediction-4CAF50?style=flat-square" />
  <img src="https://img.shields.io/badge/task-yield%20estimation-8BC34A?style=flat-square" />
  <img src="https://img.shields.io/badge/framework-OpenSTL%20extended-2196F3?style=flat-square" />
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square" />
  <img src="https://img.shields.io/badge/pytorch-supported-EE4C2C?style=flat-square" />
  <img src="https://img.shields.io/badge/status-active%20development-FF9800?style=flat-square" />
</p>

<p>
  🌱 Agricultural forecasting benchmark &nbsp; | &nbsp;
  📈 Unified training pipeline &nbsp; | &nbsp;
  🧠 Multi-model integration &nbsp; | &nbsp;
  🍅 Multi-species plant datasets
</p>

</div>

---

## 📌 Overview

AgriSTL is an agricultural spatio-temporal learning framework built upon OpenSTL and further extended for plant growth prediction and yield estimation. It is designed to provide a unified, extensible, and reproducible benchmark platform for agricultural temporal modeling from image sequences.

Different from generic spatio-temporal forecasting frameworks, AgriSTL focuses on agricultural growth processes, especially seedling-stage temporal development, where cross-day morphology changes, species-specific growth patterns, and visually subtle temporal differences make prediction more challenging. To support this setting, AgriSTL integrates multiple recent spatio-temporal prediction models into a unified pipeline and further introduces an agricultural baseline for yield-related prediction.

At the current stage, AgriSTL includes:
- multiple spatio-temporal prediction models adapted into a consistent training framework
- three plant growth prediction datasets covering Arabidopsis, tomato, and kale seedlings
- an additional Yield3DCNN baseline for yield estimation tasks
- a modular codebase suitable for future model, dataset, and task extension

---

## ✨ What makes AgriSTL different

AgriSTL is not simply a direct copy of a generic forecasting benchmark. It is tailored for agricultural temporal vision tasks with the following characteristics:

### 1. Agriculture-oriented benchmark design
AgriSTL is explicitly designed for agricultural plant growth scenarios rather than general moving-object or weather benchmarks. The framework emphasizes biological growth dynamics, visual development across days, and crop-oriented downstream usage.

### 2. Unified support for multiple recent temporal models
AgriSTL extends the base framework with a diverse set of temporal and spatio-temporal models, making it easier to compare different modeling paradigms under a shared protocol.

### 3. Plant-specific multi-species datasets
The framework includes seedling-stage datasets from three plant species:
- Arabidopsis
- Tomato
- Kale

These datasets are constructed for temporal plant growth prediction over a 20-day developmental period.

### 4. Beyond prediction: yield estimation support
In addition to future growth prediction, AgriSTL expands toward agricultural downstream tasks by incorporating Yield3DCNN as a baseline for yield estimation.

### 5. Extensible research platform
AgriSTL is intended not only as a codebase, but also as a research testbed for:
- agricultural video prediction
- temporal phenotyping
- growth dynamics modeling
- cross-species temporal learning
- yield-related visual forecasting

---

## 🧩 Supported models

AgriSTL currently supports or extends the following models:

### General temporal / spatio-temporal models
- EarthFormer
- TimesNet
- TimeMixer
- iTransformer
- DMVFN
- PredFormer
- TimeSformer
- VMRNN
- GMG

### Agricultural downstream baseline
- Yield3DCNN

These models are integrated into a unified experimental pipeline so that different architectures can be trained and evaluated under consistent settings.

---

## 🌿 Supported tasks

AgriSTL currently focuses on two major task directions.

### 1. Plant growth prediction
Given a sequence of historical plant images, the model predicts future observations during seedling growth. This setting can be used for:
- future frame prediction
- temporal morphology analysis
- growth trend modeling
- early-stage plant development forecasting

### 2. Yield estimation
AgriSTL also supports yield-related prediction through the Yield3DCNN baseline. This part aims to connect temporal representation learning with agricultural downstream analysis.

---

## 🍀 Datasets

AgriSTL includes three plant growth prediction datasets collected during the 20-day seedling stage.

### Included species
- Arabidopsis
- Tomato
- Kale

### Dataset characteristics
Each dataset consists of temporal image sequences that describe the daily growth process of seedlings. Compared with generic spatio-temporal data, these datasets are characterized by:
- subtle temporal visual changes
- strong biological growth continuity
- species-specific morphology
- agricultural interpretability

### Example applications
- future plant appearance prediction
- developmental trajectory modeling
- plant growth representation learning
- temporal benchmarking across species
- early-stage agricultural decision support

### Dataset status
If you have not yet released the dataset publicly, you can temporarily keep this section as:

> The datasets are currently being organized and will be released after paper publication or project completion.

If later you publish them, you can replace this with official download links.

---

## 🏗️ Framework structure

AgriSTL follows a modular design and can be roughly divided into the following layers:

### Core layer
Responsible for:
- training engine
- evaluation logic
- optimization utilities
- logging and checkpoint management

### Model layer
Responsible for:
- temporal forecasting models
- network definitions
- agricultural task-specific baselines

### Data layer
Responsible for:
- dataset registration
- dataloaders
- preprocessing pipeline
- train/val/test splits

### User layer
Responsible for:
- config-driven experiments
- training scripts
- testing scripts
- inference and result analysis

---

## 📂 Repository structure

A recommended repository organization is shown below. You can later adjust this block to exactly match your codebase.

```text
AgriSTL/
├── configs/                 # model and dataset configuration files
├── openstl/                 # core framework modules inherited and extended from OpenSTL
│   ├── api/
│   ├── core/
│   ├── datasets/
│   ├── methods/
│   ├── models/
│   └── modules/
├── tools/                   # training, testing, and utility scripts
├── docs/                    # project documents, figures, and notes
├── examples/                # demos or example notebooks
├── checkpoints/             # saved model weights (optional, usually ignored)
├── results/                 # evaluation outputs and logs (optional, usually ignored)
├── requirements.txt
├── environment.yml
└── README.md
