# Multilabel Scientific Keyword Classification

## Project Context

The objective is to automatically assign relevant scientific keywords (Unified Astronomy Thesaurus, UAT) to astrophysics research articles, based only on their title and abstract.  
This is a classic multilabel text classification challenge, where each article may have several associated keywords.

Automatic keyword annotation is crucial for large-scale indexing, search, and retrieval in scientific databases.

---

## Problem Statement

Given a dataset of astrophysics research papers with titles, abstracts, and verified UAT keywords, the goal is to build models that can predict the most relevant UAT keywords for new, unseen articles.

---

## Approaches

We implemented and compared two main methods:

- **Baseline:** TF-IDF vectorization + OneVsRest Logistic Regression  
  A simple, interpretable, and fast model using word frequency features.
- **Transformer:** Fine-tuning BERT (`bert-base-uncased`) for multilabel classification  
  A modern deep learning model that leverages context and semantics.

---

## Dataset

- Each data sample contains:
    - `title` (string)
    - `abstract` (string)
    - `verified_uat_labels` (list of UAT keywords)
- Data is provided as `.parquet` files (`train` and `val` sets).
- **Note:** Data files are not included in this repository. Please place them in the `data/` directory.

---

## Repository Structure

```text
.
├── data/                  # Parquet data files
├── src/                   # Source code (models, utilities)
│     └── models.py
├── main.py                # Main launcher: menu, training, evaluation
├── requirements.txt       # Python dependencies
├── report/                # Project report (PDF), figures
└── README.md
```

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/raphlp/Ma412.git
    cd Ma412
    ```

2.  Install dependencies (recommended: Python 3.8+):

    ```bash
    pip install -r requirements.txt
    ```

3.  Add your data:  
    Place your `train` and `val` `.parquet` files in the `data/` folder.


Launch the main script and follow the menu:

```bash
python main.py
```

You will be prompted to select a model:

-   1 : TF-IDF + Logistic Regression (fast, classical baseline)
-   2 : BERT (transformers, slower, contextual deep learning)

Outputs:

-   Per-label and global classification metrics (micro/macro/sample F1)
-   Example predictions for several articles


## Example Code (Baseline)

```python
from src.models import run_baseline

# Load your dataframes train, val and binarized labels Y_train, Y_val, label_names
run_baseline(train, val, Y_train, Y_val, label_names)
```

## Features

-   Modular and documented code
-   Supports classical and transformer-based multilabel classification
-   Handles imbalanced, real-world scientific data
-   Configurable sample sizes for quick testing or full-scale training
-   Saves reports and prediction examples
-   Extensible for new models and experiments

## Results (full files)

| Model                        | Micro F1 | Macro F1 | Comments                 |
| ---------------------------- | -------- | -------- | ------------------------ |
| TF-IDF + Logistic Regression | 0.31     | 0.09     | Fast, simple baseline    |
| BERT (transformers)          | 0.37     | 0.13     | Better, contextual       |

**Note**:  
The results reported above (and in the detailed report) were obtained on the **full dataset**, which required several hours of training.  
For reproducibility, the scripts fix all random seeds and allow you to rerun experiments on **smaller samples** as defined in the main script.  
Sample-based results (with the same metrics and structure) will be printed when you run the code using the default settings, but will differ  from the scores above, as those were computed on the full-scale benchmark.

Full evaluation details, error analysis, and model comparison are included in the report (`/report/`).

## Results on Small Samples

To allow for rapid testing and reproducibility, we also trained and evaluated the models on **subsamples** of the dataset (`1500` articles for training, `400` for validation). This is useful for debugging and for users without access to significant computing resources. However, these reduced-size experiments are much more challenging for multi-label classification, especially for BERT.

| Model                        | Micro F1 | Macro F1 | Comments                             |
| ---------------------------- | -------- | -------- | -------------------------------------|
| TF-IDF + Logistic Regression | 0.12     | 0.06     | More robust on small samples         |
| BERT (transformers)          | 0.05     | 0.01     | Struggles to contextualize with few data |

**Note**:  
Performance is significantly lower on small subsamples due to the high label imbalance and limited examples per class.  
The random seed fixed in the code will reproduce exactly these results on these small sample settings.
For full-scale results and analysis, see the table above and the full report (`/report/`).

## Project Report

A detailed technical report (PDF) with:

-   Problem analysis
-   Dataset exploration and statistics
-   Pipeline and algorithm descriptions
-   Experiments and hyperparameters
-   Results, metrics, and plots
-   Discussion and limitations

Available in `/report/`.

---

## Author

Raphaël Laupies  
IPSA, MA412, 2025

---
