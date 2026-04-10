# AgriSTL

AgriSTL is a unified spatio-temporal learning framework for agricultural growth prediction and yield estimation. Built upon OpenSTL, this project extends the original framework by integrating a diverse set of spatio-temporal forecasting models and introducing agricultural vision benchmarks tailored for plant growth modeling.

The current version of AgriSTL supports multiple temporal and spatio-temporal prediction architectures, including EarthFormer, TimesNet, TimeMixer, iTransformer, DMVFN, PredFormer, TimeSformer, VMRNN, and GMG. In addition, we extend the framework with a Yield3DCNN baseline for yield estimation tasks. To facilitate benchmarking in agricultural scenarios, we further construct three plant growth prediction datasets covering 20-day seedling-stage image sequences of Arabidopsis, tomato, and kale.

## 1. Highlights

- Unified framework for agricultural spatio-temporal prediction
- Built on top of OpenSTL with extended model support
- Supports both growth forecasting and yield estimation tasks
- Includes three plant seedling growth datasets
- Covers multiple recent temporal and spatio-temporal models in a unified training pipeline
- Provides a flexible interface for adding new datasets, models, and evaluation protocols

## 2. Motivation

Spatio-temporal learning has achieved remarkable progress in generic forecasting benchmarks, yet its application to agricultural growth modeling remains limited. Agricultural scenarios involve long-term developmental dynamics, cross-day visual variations, species-specific growth patterns, and task-specific objectives such as future growth prediction and yield estimation. Existing generic frameworks are often not designed for these settings.

AgriSTL aims to bridge this gap by providing a unified and extensible benchmark framework for agricultural spatio-temporal learning. It enables consistent training, evaluation, and comparison across different architectures and datasets, thereby supporting the development of robust predictive models for plant phenotyping and smart agriculture.

## 3. Supported tasks

AgriSTL currently supports the following tasks:

### 3.1 Plant growth prediction

Given historical image sequences of plant seedlings, the model predicts future growth observations over time. This task can be used to study temporal growth dynamics, developmental trends, and future plant morphology changes.

### 3.2 Yield estimation

AgriSTL also includes a yield estimation task with the Yield3DCNN baseline. This branch is designed for learning from agricultural visual data to estimate crop yield-related targets.

## 4. Supported models

The framework currently includes or has been extended to support the following models:

- EarthFormer
- TimesNet
- TimeMixer
- iTransformer
- DMVFN
- PredFormer
- TimeSformer
- VMRNN
- GMG
- Yield3DCNN

Notes:
- Some models are adapted from their original implementations into the AgriSTL training and evaluation pipeline.
- Different models may correspond to different input assumptions, temporal modeling strategies, or task settings.
- Detailed configuration files are expected to be placed under the `configs/` directory.

## 5. Datasets

AgriSTL includes three plant growth prediction datasets collected during the 20-day seedling stage:

- Arabidopsis
- Tomato
- Kale

### 5.1 Dataset characteristics

Each dataset consists of temporal image sequences that describe the growth process of seedlings across multiple days. These datasets are designed to support spatio-temporal prediction research in agricultural scenarios.

Potential applications include:

- future frame prediction of plant growth
- temporal plant development analysis
- comparative evaluation across species
- agricultural representation learning
- early-stage growth monitoring

### 5.2 Dataset organization

Please organize the datasets in a structure similar to the following:

```text
data/
├── Arabidopsis/
│   ├── train/
│   ├── val/
│   └── test/
├── Tomato/
│   ├── train/
│   ├── val/
│   └── test/
└── Kale/
    ├── train/
    ├── val/
    └── test/
