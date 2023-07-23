# The Role of Relevance in Fair Ranking <!-- omit in toc -->
An examination of the role of relevance in fair exposure-based ranking. For more details please see our [SIGIR 2023 paper](https://doi.org/10.1145/3539618.3591933).

## Contents <!-- omit in toc -->
- [Setting Up](#setting-up)
  - [1. Environment and Prerequisites](#1-environment-and-prerequisites)
  - [2. Obtaining the Data](#2-obtaining-the-data)
- [Main Experimental Grid](#main-experimental-grid)
  - [1. Running Experiments](#1-running-experiments)
  - [2. Aggregating Results](#2-aggregating-results)
- [Auxiliary Experiments](#auxiliary-experiments)
  - [Fair Ranking Intervention](#law-school-admissions-simulation)
  - [Imbalanced Groups](#group-information-in-feature-representations)
- [Citation](#citation)


## Setting Up
### 1. Environment and Prerequisites
Run the following commands to clone this repo and create the Conda environment:

```
git clone git@github.com:AparnaB/role-of-relevance-in-fair-ranking.git
cd role-of-relevance-in-fair-ranking/
conda env create -f environment.yml
conda activate rel_exp
```

### 2. Obtaining the Data
We provide `Rnormal.csv` and `Rpareto.csv` of the synthetic datasets in the `data` folder. We also provide notebooks for processing all data in the `notebooks/` folder.


## Main Experimental Grid
### 1. Running Experiments
To reproduce the experiments in the paper which involve training ranking models using pretrained click and propensity models, run the `main.py` file with different dataset and experimental settings.

Sample bash scripts showing the command can be found in `bash_scripts/`.

### 2. Aggregating Results
We aggregate results and generate tables using scripts in the `lib` folder.

## Auxiliary Experiments
### Fair Ranking Intervention
We provide script used simulate fair re-ranking interventions in `lib/fair_reranking.py`.

### Imbalanced Groups
To reproduce the imbalanced groups experiment described in Section 6.3 of the paper, run the `lib/get_imbalanced_fairness.py` script.

## Citation
If you use this code in your research, please cite the following publication:
```
@article{balagopalan2023,
  title={The Role of Relevance in Fair Ranking},
  author={Balagopalan, Aparna and Jacobs, Z. Abigail and Biega, Asia},
  conference={SIGIR},
  year={2023}
}

```
