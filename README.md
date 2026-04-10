<div align="center">

# AgriSTL
### An Agricultural Spatio-Temporal Learning Framework for Plant Growth Prediction

<p>
  <img src="https://img.shields.io/badge/task-plant%20growth%20prediction-43A047?style=flat-square" />
  <img src="https://img.shields.io/badge/task-yield%20estimation-7CB342?style=flat-square" />
  <img src="https://img.shields.io/badge/task-weather%20forecasting-1E88E5?style=flat-square" />
  <img src="https://img.shields.io/badge/framework-OpenSTL%20extended-3949AB?style=flat-square" />
  <img src="https://img.shields.io/badge/status-active%20development-F9A825?style=flat-square" />
  <img src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square" />
</p>

<p>
  🌱 Plant growth prediction &nbsp; | &nbsp;
  🍅 Agricultural temporal modeling &nbsp; | &nbsp;
  ⛅ Weather-aware forecasting &nbsp; | &nbsp;
  📈 Unified benchmark framework
</p>

[📘 Overview](#overview) |
[📰 News](#news-and-updates) |
[🛠️ Installation](#installation) |
[🚀 Getting Started](#getting-started) |
[🧠 Supported Methods](#supported-methods) |
[🌿 Datasets and Tasks](#datasets-and-tasks) |
[🎞️ Visualization](#visualization) |
[🙏 Acknowledgement](#acknowledgement)

</div>

---

## Introduction

AgriSTL is an agricultural spatio-temporal learning framework built upon OpenSTL and further extended for agricultural predictive tasks. It is designed to provide a unified, extensible, and reproducible benchmark for temporal modeling from agricultural observations, with a primary focus on plant growth prediction.

Compared with generic spatio-temporal learning benchmarks, AgriSTL places greater emphasis on agricultural growth dynamics, especially the subtle and continuous developmental changes observed in seedling-stage plant image sequences. In addition to plant growth prediction as the core task, the framework also supports two other representative agricultural task directions, namely yield estimation and weather forecasting, forming a unified benchmark that covers three typical agricultural temporal prediction scenarios.

At the current stage, AgriSTL integrates multiple recent temporal and spatio-temporal prediction models into a common training and evaluation pipeline, and provides plant-oriented benchmark datasets covering Arabidopsis, Tomato, and Kale during the 20-day seedling stage.

---

## Overview

AgriSTL is developed to bridge the gap between general spatio-temporal learning frameworks and real agricultural prediction needs.

The framework is mainly centered on plant growth prediction, while also maintaining compatibility with agricultural weather forecasting and yield-related analysis. This design enables researchers to study temporal modeling methods in a more application-oriented agricultural context.

AgriSTL currently provides:

- a unified codebase for agricultural spatio-temporal learning
- integrated support for multiple recent temporal prediction models
- benchmark datasets for seedling-stage plant growth prediction
- support for three representative agricultural task types:
  - plant growth prediction
  - yield estimation
  - weather forecasting
- a modular pipeline for future extension to new datasets, models, and downstream tasks

Among these tasks, plant growth prediction is the major focus of the current project, while yield estimation and weather forecasting are included as complementary task directions for broader agricultural applicability.

---

## Key Features

- Agriculture-oriented benchmark design for temporal prediction
- Main focus on plant growth prediction from image sequences
- Unified support for multiple recent spatio-temporal learning models
- Coverage of three representative agricultural task types
- Extensible framework for new models, datasets, and evaluation protocols
- Built upon the solid foundation of OpenSTL and adapted for agricultural applications

---

## News and Updates

- [2026-04-10] AgriSTL project repository is created.
- [2026-04-10] Initial version of the README is released.
- [Coming Soon] Benchmark datasets, checkpoints, and more detailed experiment documents will be released gradually.

---

## Datasets and Tasks

AgriSTL currently covers three representative agricultural task directions.

### 1. Plant growth prediction

This is the primary task of AgriSTL. Given historical plant observations, the model predicts future growth states. This task is designed for studying temporal plant development, morphological evolution, and growth-aware prediction in agricultural scenarios.

The current benchmark datasets focus on 20-day seedling-stage growth image sequences of:

- Arabidopsis
- Tomato
- Kale

### 2. Yield estimation

AgriSTL additionally supports yield-related prediction as an agricultural downstream task. This direction is currently included as an extension of the framework and will be gradually enriched in future releases.

### 3. Weather forecasting

AgriSTL also supports weather forecasting as another representative agricultural temporal task. This task is intended to improve the framework’s broader applicability in agricultural spatio-temporal prediction research.

### Dataset status

The plant growth datasets are currently being organized for public release. More complete documentation and access instructions will be provided in future updates.

---

## Supported Methods

AgriSTL extends the OpenSTL framework by integrating multiple representative temporal and spatio-temporal prediction models into a unified experimental pipeline.

### Currently supported methods

- √ IncepU (SimVP.V1) (CVPR'2022)
- √ gSTA (SimVP.V2) (arXiv'2022)
- √ TAU (CVPR'2023)
- √ EarthFormer (NeurIPS'2022)
- √ TimesNet (ICLR'2023)
- √ TimeMixer (ICLR'2024)
- √ iTransformer (ICLR'2024)
- √ DMVFN (CVPR'2023)
- √ TimeSformer (ICML'2021)
- √ GMG
- √ PhyDNet (CVPR'2020)
- √ MIM (CVPR'2019)
- √ PredRNNv2
- √ MMVP (ICCV'2023)
- √ PredFormer
- √ Yield3DCNN (AgriSTL baseline)

### Internal methods in this project

- RDMN (AgriSTL internal method, currently unpublished)
- DSAVEN (AgriSTL internal method, currently unpublished)

### Notes

- Some methods are adapted and re-organized under the AgriSTL training and evaluation pipeline.
- RDMN and DSAVEN are currently internal methods of this project and have not yet been formally published or open-sourced.
- Additional temporal prediction baselines and corresponding configurations will be gradually released in future updates.

---

## Repository Structure

A typical repository structure is as follows:

```text
AgriSTL/
├── configs/                  # configuration files for datasets and models
├── openstl/                  # core framework modules inherited and extended from OpenSTL
│   ├── api/
│   ├── core/
│   ├── datasets/
│   ├── methods/
│   ├── models/
│   └── modules/
├── tools/                    # training, testing, and utility scripts
├── docs/                     # project documents and figures
├── checkpoints/              # pretrained checkpoints (to be released)
├── results/                  # experiment outputs and logs
├── RUN_SCRIPTS.md            # example commands for running experiments
├── requirements.txt
├── environment.yml
└── README.md
