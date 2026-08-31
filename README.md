# Automated Thyroid Disease Detection Using a Hybrid Autoencoder–TabTransformer–MLP Model

Automated thyroid disease detection using a hybrid deep learning model combining an Autoencoder-based numerical feature encoder, TabTransformer, and Multi-Layer Perceptron (MLP) for structured clinical and biochemical data.

## Overview

Thyroid disorders are common endocrine diseases that require analysis of multiple clinical and biochemical parameters for diagnosis.

This project presents a hybrid **Autoencoder–TabTransformer–MLP** architecture for binary thyroid disease classification using structured clinical data.

The proposed model processes numerical and categorical features through two complementary branches:

- **Numerical Encoder:** Learns nonlinear representations from numerical clinical and biochemical features.
- **TabTransformer:** Learns contextual relationships among categorical clinical features using self-attention.
- **MLP Classifier:** Combines the learned representations and performs the final binary classification.

The overall workflow is:

```text
                 Thyroid Clinical Dataset
                          |
             +------------+------------+
             |                         |
      Numerical Features       Categorical Features
             |                         |
     Numerical Encoder         Categorical Embeddings
        32 → 16 units                   |
             |                   TabTransformer
             |                  2 Transformer Blocks
             |                         |
             +------------+------------+
                          |
                   Feature Fusion
                          |
                    MLP Classifier
                    128 → 64 units
                          |
                       Sigmoid
                          |
              Thyroid Disease Prediction
```

## Dataset

The experiments use a publicly accessible thyroid disease patient dataset containing:

- **3,772 patient records**
- **28 total columns**
- **27 predictor variables**
- **1 target variable (`Class`)**

### Dataset Source

The dataset is available on Kaggle:

👉 **[Download the Thyroid Disease Patient Dataset](https://www.kaggle.com/datasets/kapoorprakhar/thyroid-disease-patient-dataset)**

### Target Distribution

| Class | Description | Samples |
|---|---|---:|
| 0 | Healthy | 3,541 |
| 1 | Thyroid Disease | 231 |

> **Note:** The dataset is not included in this repository. Please download it from Kaggle before running the notebook.

## Features

The 27 predictor variables are divided into numerical and categorical/binary features.

### Numerical Features

- Age
- TSH
- T3
- TT4
- T4U
- FTI

### Categorical / Binary Features

- Sex
- Referral source
- On thyroxine
- Query on thyroxine
- On antithyroid medication
- Sick
- Pregnant
- Thyroid surgery
- I131 treatment
- Query hypothyroid
- Query hyperthyroid
- Lithium
- Goitre
- Tumor
- Hypopituitary
- Psych
- TSH measured
- T3 measured
- TT4 measured
- T4U measured
- FTI measured

## Data Preprocessing

To prevent information leakage, the original dataset is first divided into training, validation, and independent test subsets.

```text
Training   : 60%
Validation : 20%
Testing    : 20%
```

A fixed random seed of `42` is used for the main experiment.

All data-dependent preprocessing parameters are estimated exclusively from the training subset and then applied unchanged to the validation and test subsets.

The preprocessing pipeline includes:

1. Conversion of TSH values to numerical form.
2. Missing numerical-value imputation.
3. Missing categorical-value imputation.
4. IQR-based outlier detection.
5. Categorical encoding.
6. Numerical feature standardization.

### Outlier Handling

The IQR method is applied to:

- Age
- TSH
- T3
- TT4
- T4U
- FTI

Outlier removal is performed only on the training data.

After training-set outlier removal:

```text
Training samples   : 1,790
Validation samples : 755
Test samples       : 755
```

## Proposed Model

### 1. Numerical Encoder

The six numerical features are processed through two fully connected layers:

```text
Input
  ↓
Dense(32) + ReLU
  ↓
Dense(16) + ReLU
```

The numerical encoder learns nonlinear representations of the demographic and biochemical features.

> In this implementation, the numerical branch is used as an encoder-derived representation within the supervised classifier. No separate unsupervised autoencoder pretraining or reconstruction loss is used.

### 2. TabTransformer

The 21 categorical/binary features are converted into embedding vectors and processed using a TabTransformer-style architecture.

Configuration:

```text
Embedding dimension : 16
Attention heads     : 2
Key dimension       : 16
Transformer blocks  : 2
```

The transformer learns contextual relationships among categorical clinical attributes.

### 3. Feature Fusion

The numerical and categorical representations are concatenated:

```text
Numerical Representation
          +
Categorical Representation
          |
          ↓
    Feature Fusion
```

### 4. MLP Classifier

The fused representation is passed through an MLP:

```text
Dense(128) + ReLU
Batch Normalization
Dropout(0.3)

Dense(64) + ReLU
Batch Normalization
Dropout(0.3)

Dense(1) + Sigmoid
```

The final sigmoid layer produces the probability of thyroid disease.

## Training Configuration

| Parameter | Value |
|---|---|
| Python | 3.10 |
| TensorFlow / Keras | 2.15.0 |
| NumPy | 1.26.4 |
| scikit-learn | 1.4.2 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Loss Function | Binary Cross-Entropy |
| Batch Size | 64 |
| Maximum Epochs | 55 |
| Dropout Rate | 0.3 |
| Activation Functions | ReLU, Sigmoid |
| Train/Validation/Test Split | 60:20:20 |
| Random Seed | 42 |
| LR Reduction Factor | 0.5 |
| LR Scheduler Patience | 5 epochs |
| Minimum Learning Rate | 1e-6 |
| Classification Threshold | 0.5 |

Class weights are computed from the training data to account for class imbalance.

## Evaluation Metrics

The model is evaluated using an independent test set.

The following metrics are used:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Reported Results

On the independent test set, the proposed model achieved:

| Metric | Result |
|---|---:|
| Accuracy | **98.01%** |
| Weighted Precision | **97.99%** |
| Weighted Recall | **98.01%** |
| Weighted F1-score | **98.00%** |
| ROC-AUC | **0.98** |

### Class-wise Performance

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Healthy | 0.9887 | 0.9901 | 0.9894 |
| Thyroid Disease | 0.8444 | 0.8261 | 0.8352 |

The minority thyroid-disease class has lower recall than the healthy class, reflecting the class imbalance present in the dataset.

## Robustness Evaluation

The experimental procedure is additionally repeated across five stratified train-validation-test partitions using the following random seeds:

```text
42
52
62
72
82
```

The reported mean performance is:

| Metric | Mean ± SD |
|---|---:|
| Accuracy | **98.01 ± 0.11%** |
| Weighted Precision | **97.99 ± 0.11%** |
| Weighted Recall | **98.01 ± 0.11%** |
| Weighted F1-score | **98.00 ± 0.11%** |
| ROC-AUC | **0.981 ± 0.001** |


## Installation

Clone the repository:

```bash
git clone https://github.com/shibithakp3/thyroid-disease-detection.git
cd thyroid-disease-detection
```

Create a virtual environment:

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The main software environment used for the experiments is:

```text
tensorflow==2.15.0
numpy==1.26.4
scikit-learn==1.4.2
pandas
matplotlib
seaborn
```

## Running the Project

The main implementation is available in:

```text
thyroid_Hybrid_model.ipynb
```

The notebook can be executed using:

- Google Colab
- Jupyter Notebook
- JupyterLab

### Google Colab

1. Open `thyroid_Hybrid_model.ipynb` in Google Colab.
2. Obtain the required thyroid dataset from the original public source.
3. Upload the dataset or configure the dataset path.
4. Run the notebook cells sequentially.
5. Review the training curves and evaluation results.

## Reproducibility

The project follows a leakage-aware experimental procedure.

The following settings are fixed for reproducibility:

- Stratified data partitioning
- Random seeds
- Training-only preprocessing
- Training-only outlier removal
- Training-derived imputation parameters
- Training-derived categorical mappings
- Training-derived scaling parameters
- Fixed model architecture
- Fixed training hyperparameters
- Independent test-set evaluation

The repeated experiments use:

```text
42, 52, 62, 72, 82
```

## Limitations

This project is a research implementation based on a single historical public thyroid dataset.

The reported results should not be interpreted as clinical diagnostic performance.
