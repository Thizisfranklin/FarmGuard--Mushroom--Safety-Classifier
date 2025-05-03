# Project Title

Mushroom Classification Challenge

## One Sentence Summary

This repository holds an attempt to classify mushrooms as edible or poisonous based on various physical characteristics using Random Forest and Naive Bayes models on the Kaggle Mushroom Classification dataset ([https://www.kaggle.com/uciml/mushroom-classification](https://www.kaggle.com/uciml/mushroom-classification)).

## Overview

**Task Definition:** The task defined by the Kaggle challenge is to use various categorical features of mushroom samples—such as cap shape, odor, gill size, and stalk surface characteristics—to predict whether a mushroom is edible (`e`) or poisonous (`p`).

**Approach:**
We formulated this as a binary classification problem. After encoding categorical variables and handling missing data, we compared the performance of two algorithms:

1. **Random Forest Classifier** – ensemble of decision trees.
2. **Gaussian Naive Bayes** – probabilistic model assuming feature independence.

**Performance Summary:**
Our best model (Random Forest) achieved near-perfect classification performance on the held-out test set, with precision, recall, and F1-score all exceeding 0.99.

## Summary of Work Done

* Data loading and exploratory analysis
* Data cleaning: handled missing stalk-root values, encoded categorical features
* Feature selection: removed redundant and low-importance features
* Model training and evaluation: compared Random Forest vs. Naive Bayes
* Generated submission file for Kaggle challenge

## Data

* **Type:** CSV file (`mushrooms.csv`) of categorical features.
* **Input:** 22 features (e.g., cap shape, odor, gill size) encoded as single characters.
* **Output:** Edible (`e`) or Poisonous (`p`) label.
* **Size:** 8,124 samples.
* **Split:** 80% training (6,499 samples), 10% validation (812 samples), 10% test (813 samples).

### Preprocessing / Cleanup

* Converted target and selected features to numeric labels.
* One-hot encoded high-cardinality categorical features (`odor`, `gill-color`, etc.) using `pd.get_dummies`.
* Dropped the `stalk-root` feature due to extensive missing values (`?`).

## Data Visualization

* Crosstabulations for each feature against the class label.
* Histograms and bar plots highlighting distribution differences (e.g., certain odors correlated strongly with poisonous labels).



## Problem Formulation

* **Input:** Preprocessed feature matrix (`X`) with one-hot and label-encoded variables.
* **Output:** Binary label (`y`): 0 = edible, 1 = poisonous.
* **Models:**

  * Random Forest Classifier
  * Gaussian Naive Bayes
* **Hyperparameters:**

  * Random Forest: `n_estimators=100`, `max_depth=None`, `random_state=42`
  * Naive Bayes: default parameters

## Training

* **Environment:** Python 3.8, scikit-learn 1.0, run on a standard CPU machine.
* **Duration:** \~10 minutes for Random Forest on 6,499 samples.
* Training and validation curves plotted to monitor overfitting.
* Early stopping based on validation accuracy plateau.

## Performance Comparison

* **Metrics:** Accuracy, precision, recall, F1-score.

| Model                | Train Acc | Val Acc | Test Acc |
| -------------------- | --------- | ------- | -------- |
| Random Forest        | 1.00      | 1.00    | 1.00     |
| Gaussian Naive Bayes | 0.98      | 0.98    | 0.98     |

* **ROC Curves:** Plotted for both models showing near-perfect AUC.

## Conclusions

* Random Forest outperformed Naive Bayes, achieving perfect classification on the test set.
* High feature separability in the mushroom dataset makes this classification problem relatively straightforward.

## Future Work

* Experiment with other tree-based ensembles (e.g., XGBoost).
* Investigate feature importance further and conduct dimensionality reduction.
* Deploy model to a web service for real-time classification app.

## How to Reproduce Results

1. Clone this repository.
2. Install requirements: `pip install -r requirements.txt`.
3. Download `mushrooms.csv` from [Kaggle Mushroom Classification](https://www.kaggle.com/uciml/mushroom-classification).
4. Run `data_preprocessing.ipynb`, `feature_visualization.ipynb`, `model_training.ipynb`.

## Directory Structure

```
├── data/
│   └── mushrooms.csv
├── notebooks/
│   ├── data_preprocessing.ipynb
│   ├── feature_visualization.ipynb
│   └── model_training.ipynb
├── src/
│   ├── utils.py
│   ├── preprocess.py
│   └── models.py
├── submission.csv
├── requirements.txt
└── README.md
```

**File Descriptions:**

* `utils.py`: helper functions for data loading and plotting.
* `preprocess.py`: scripts for data cleaning and encoding.
* `models.py`: functions to build and evaluate models.
* `model_training.ipynb`: trains Random Forest and Naive Bayes models.
* `submission.csv`: example Kaggle submission file.

## Software Setup

* Python 3.8+
* `pip install numpy pandas scikit-learn matplotlib`

## Citations

* Kaggle Mushroom Classification dataset: [https://www.kaggle.com/uciml/mushroom-classification](https://www.kaggle.com/uciml/mushroom-classification)
