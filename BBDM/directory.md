directory.md — GEETANJALI_PHD_IRIS_PAD/BBDM

Overview

This repository (BBDM) supports Geetanjali's PhD work on IRIS Presentation Attack Detection (PAD). It contains data preparation, model definitions, training/evaluation scripts, configuration files, and resources/results from experiments.

Files and folders

- LICENSE: Project license and usage terms.
- readme.md: Project overview and quick start instructions.
- environment.yml: Conda environment specification listing required packages.
- Register.py: Script for dataset or experiment registration (entrypoint for registrations).
- main.py: Primary entrypoint to run training, evaluation, or experiments.
- preprocess_and_evaluation.py: Combined preprocessing and evaluation utilities/scripts.
- utils.py: General helper functions used across scripts.

Directories

- __pycache__/ : Python bytecode caches (auto-generated).
- configs/ : Configuration files (YAML/JSON) describing experiments, hyperparameters, and paths.
- datasets/ : Raw and processed dataset files, dataset loaders, and dataset-building scripts.
- model/ : Model architectures, layers, and model-related utilities.
- evaluation/ : Evaluation metrics, scripts, and tools to analyze model performance.
- resources/ : External assets, pretrained weights, or reference data used by experiments.
- results/ : Output from experiments: logs, saved models, metrics, and plots.
- runners/ : Scripts to run experiments, training loops, or batch jobs.
- shell/ : Shell helper scripts (bash wrappers, automation scripts).

Notes

- Read readme.md and configs/ before running experiments to set paths and parameters.
- Use environment.yml to create a reproducible conda environment.
- This file is a brief summary; inspect individual files and folders for implementation details.
