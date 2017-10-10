# Task-Specific Scoring Functions for Predicting Ligand Binding Poses and Affinity and for Screening Enrichment

Molecular modeling has become an essential tool to assist in early stages of drug discovery and development. Molecular docking, scoring, and virtual screening are three such modeling tasks of particular importance in computer-aided drug discovery. They are used to computationally simulate the interaction between small drug-like molecules, known as *ligands*, and a target protein whose function is to be modulated. Scoring functions (SFs) are typically employed to predict the binding conformation (*docking task*),
binding affinity (*scoring task*), and binary activity level (*screening task*) of ligands against a critical protein target in a disease's pathway. In most molecular docking software packages available today, a generic binding affinity-based (BA-based) SF is invoked for all three tasks to solve three different, but related, prediction problems. The limited predictive accuracies of such SFs in these three tasks have been a major roadblock toward cost-effective drug discovery. 

# Task-Specific Scoring Functions
This repository provides Python scripts for building task-specific scoring functions that significantly improve upon the performance of binding-affinity-based approaches. The three scoring functions are:
- **BT-Score**: an ensemble ML SF based on boosted decision trees and thousands of descriptors to predict BA. BT-Score's predictions for out-of-sample test complexes have 0.825 correlation with experimental BA. This represents more than 14% and 31% improvement over the performance of the ML SF RF-Score and empirical model X-Score whose Pearson's correlation coefficients are 0.725 and 0.627, respectively. Despite its high accuracy in the scoring task, we find the docking and screening performance of BT-Score is far from ideal which motivated us to develop the following task-specific SFs.
- **BT-Dock**: a boosted-tree docking model fitted to a large number of native and computer-generated ligand conformations and then optimized to predict binding poses explicitly. The model has shown an average of 25% improvement over its BA-based counterparts in different pose prediction scenarios.
- **BT-Screen**: a screening SF that directly models ligand activity as a classification problem. BT-Screen is fitted to thousands of active and inactive protein-ligand complexes to optimize it for finding real actives from databases of ligands not seen in its training set. Our results suggest that BT-Screen can be 73% more accurate than the best conventional SF in enriching datasets of active and inactive ligands for many protein targets.

# Download & Installation
The scripts provided here are written in Python. Therefore, Python 3.X or 2.7 must be available on the machine (Linux) to run them. The following modules are required before running the script:
- Numpy (version 1.11.1)
- Scipy (0.18.1)
- Pandas (0.18.1)
- Scikit-learn (0.18.1)
- XGBoost (0.6).

Then [download the project](https://github.com/ashtawy/task-specific_scoring_functions/archive/master.zip), unzip it, and go to its directory (*$cd /path/to/task-specific_scoring_functions/*) in order to train and test task-specific scoring functions. 

# Train and Test Task-Specific & Conventional Scoring Functions
With one command, you could train and test a task-specific (BT-Score, BT-Dock, and BT-Screen) or generic scoring functions (RF-Score and X-Score). For example, the following command shows how to build BT-Score for the ligand scoring task (predicting binding affinity):
```bash 
python train_test_task-specific_sfs.py --sfname BT-Score --task score

Performance of BT-Score on the scoreing task:
N_Training  N_Test  N_Descriptors     Rp     Rs     SD   RMSE
      3000     195           2714  0.827  0.813  1.318  1.311
```

You could as easily train RF-Score and test it on the screening task:
```bash
python train_test_task-specific_sfs.py --sfname RF-Score --task screen

Performance of RF-Score on the screening task:
N_Training  N_Test  N_Descriptors    EF1    EF5  EF10
      3000   12675            216  4.103  2.256  1.59
```

... and compare it against the task-specific BT-Screen on the same task:
```bash
python train_test_task-specific_sfs.py --sfname BT-Screen --task screen

Performance of BT-Screen on the screening task:
N_Training  N_Test  N_Descriptors     EF1    EF5   EF10
      3000   12675           2714  33.901  10.821  5.59
```

# Molecular datasets
Our training and validation datasets are obtained from [PDBbind (versions 2007 & 2014)](http://www.pdbbind.org.cn/). We performed docking using Autodock Vina and extracted molecular descriptors for each protein-ligand complex. The processed datasets are included the directory data:

+   [data](./data)
    +   [input](./data/input)
        *   [dock](./data/input/dock)
        *   [screen](./data/input/screen)
        *   [score](./data/input/score)

# Contributors
* [Hossam M. Ashtawy](http://www.ashtawy.com)
* [Nihar R. Mahapatra](http://www.egr.msu.edu/~nrm)


