# Tailoring Data Preprocessing to Enhance the Predictive Performance of Ensemble-based AutoML

This repository provides the prototypical implementation used for our paper "Tailoring Data Preprocessing to Enhance the Predictive Performance of Ensemble-based AutoML".
The prototype tailors a preprocessing search space to the data characteristics of a given dataset to reduce the search space size.
The data characteristics currently supported are a small data set size, high dimensionality and class imbalance.
The search space is adapted as follows:
- Small Data (<=5000 data instances): Oversampling (Random Oversampling, Smote) and Feature Selection (Feature Importance, Mutual Information)
- High Dimensionality (>100 features): Feature Selection (Feature Importance, Mutual Information) and Feature Reduction (PCA, LDA)
- Class Imbalance: Oversampling (Random Oversampling, Smote) and Undersampling (Edited-Nearest-Neighbours, Tomek Links)

If multiple, i.e., more than one, data characteristics are present in a dataset, the search space consists of all of the respective preprocessing methods.
The adapted search space is then passed to the Auto-CEn[1] framework, which optimizes the preprocessing methods together with classifiers to create an optimized classifier ensemble.
Further information can be found in our paper "Tailoring Data Preprocessing to Enhance the Predictive Performance of Ensemble-based AutoML".

[1] Julius Voggesberger et al. 2025. Auto-CEn: AutoML for Classifier Ensembles - Diversity-based Classifier Selection and Decision Fusion Optimization. 
2025 IEEE 12th International Conference on Data Science and Advanced Analytics (DSAA). 

## Installation
To use the framework Python 3.10 and Ubuntu >=22.04 are required.
The dependencies are specified in 'auto_cen/requirements.txt'.

## Running the Prototype
To run the prototype, execute the example in 'example.py'.
Alternatively, the script used for the evaluation can be found in 'Evaluation/Experiments/run_auto_cen_dp.py'.
To execute the script simply call `python run_auto_cen_dp.py openml_id model_budget seed`,
where 
- openml_id: The OpenML ID of the dataset.
- model_budget: The number of preprocessing+classifier configurations to evaluate in the optimization.
- seed: The random seed to make the run reproducible.


## Reproducibility
This repository provides the measurements and code to reproduce the results of the paper "Tailoring Data Preprocessing to Enhance the Predictive Performance of Ensemble-based AutoML".
The results of the evaluation can be found in 'Evaluation/Results/'.
The code for running the experiments can be found in 'Evaluation/Experiments/'.
To execute the experiments, first run `create_evaluation_script.py` to generate a shell script containing all commands to be executed.
The shell script calls `run_auto_cen_dp.py` with different arguments, i.e., to evaluate different datasets.
The code for recreating the tables and figures of the paper can be found in 'Evaluation/Visualization Code'.
To recreate the tables and figures, simply run the jupyter notebook `tables_and_figures.ipynb`.

