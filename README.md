<div align="center">

<!-- 可替换为你的项目主图 -->
<!-- <img src="docs/assets/agristl_logo.png" width="38%"> -->

# AgriSTL: An Agricultural Spatio-Temporal Learning Framework

<p align="center">
  <img src="https://img.shields.io/badge/Task-Plant%20Growth%20Prediction-43A047?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Yield%20Estimation-7CB342?style=flat-square" />
  <img src="https://img.shields.io/badge/Task-Weather%20Forecasting-1E88E5?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-OpenSTL%20Extended-3949AB?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-Active%20Development-F9A825?style=flat-square" />
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square" />
</p>

<p align="center">
  🌱 Plant growth prediction &nbsp; | &nbsp;
  🍅 Agricultural temporal learning &nbsp; | &nbsp;
  ⛅ Weather-aware forecasting &nbsp; | &nbsp;
  📈 Unified benchmark framework
</p>

[📘 Overview](#overview) |
[🛠️ Installation](#installation) |
[🚀 Getting Started](#getting-started) |
[🧠 Supported Methods](#overview-of-supported-methods-and-agricultural-tasks) |
[🎞️ Visualization](#visualization) |
[🆕 News](#news-and-updates) |
[🙏 Acknowledgement](#acknowledgement)

</div>

## Introduction

AgriSTL is an agricultural spatio-temporal learning framework built upon OpenSTL and further extended for agricultural predictive tasks. It is designed to provide a unified, extensible, and reproducible benchmark for temporal modeling from agricultural observations, with plant growth prediction as its primary focus.

Compared with generic spatio-temporal learning benchmarks, AgriSTL places greater emphasis on agricultural growth dynamics, especially the subtle and continuous developmental changes observed in seedling-stage plant image sequences. In addition to plant growth prediction as the core task, the framework also supports two complementary agricultural task directions, namely yield estimation and weather forecasting, thereby covering three representative agricultural temporal prediction scenarios within a common experimental pipeline.

At the current stage, AgriSTL integrates multiple recent temporal and spatio-temporal prediction models, and provides plant-oriented benchmark datasets covering Arabidopsis, Tomato, and Kale during the seedling stage. Among these directions, plant growth prediction remains the major focus of the current project.

<p align="center">
  <img src="docs/assets/agristl_framework.png" width="88%">
</p>

<p align="center">
  Overall framework of AgriSTL. The figure can be replaced with your project overview illustration.
</p>

<p align="right">(<a href="#top">back to top</a>)</p>

## Overview

<details open>
<summary>Main Features and Plans</summary>

- <b>Agriculture-oriented benchmark design.</b>
  AgriSTL is designed for agricultural temporal learning rather than generic spatio-temporal benchmarks. It focuses particularly on plant growth prediction from image sequences, while also maintaining compatibility with yield estimation and weather forecasting.

- <b>Unified spatio-temporal learning pipeline.</b>
  AgriSTL extends the OpenSTL framework by integrating multiple representative temporal and spatio-temporal models into a common training and evaluation pipeline, making comparisons across methods more convenient and reproducible.

- <b>Plant-focused benchmark datasets.</b>
  The current benchmark includes seedling-stage growth datasets of Arabidopsis, Tomato, and Kale, aiming to support temporal plant development modeling in agricultural scenarios.

- <b>Future plans.</b>
  We plan to continue releasing benchmark datasets, experiment configurations, pretrained checkpoints, visualization resources, and additional baselines for agricultural temporal predictive learning.

</details>

<details open>
<summary>Code Structure</summary>

- `openstl/api` contains experiment runners and interfaces.
- `openstl/core` contains training utilities, optimization components, and metrics.
- `openstl/datasets` contains datasets and dataloaders.
- `openstl/methods` contains training methods for different predictive models.
- `openstl/models` contains the main network architectures.
- `openstl/modules` contains reusable layers and blocks.
- `configs/` contains task-specific and model-specific configuration files.
- `tools/` contains executable scripts for training, testing, and related utilities.
- `RUN_SCRIPTS.md` contains example running commands for experiments.

</details>

<p align="right">(<a href="#top">back to top</a>)</p>

## News and Updates

[2026-04-10] AgriSTL project repository is created.

[2026-04-10] Initial README and project structure are organized.

[Coming Soon] Benchmark datasets, checkpoints, and more detailed experimental resources will be released gradually.

## Installation

AgriSTL can be installed in a conda environment.

```shell
git clone https://github.com/LYQ1107/AgriSTL.git
cd AgriSTL
conda env create -f environment.yml
conda activate AgriSTL
python setup.py develop
