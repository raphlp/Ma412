# main.py

import random
import numpy as np
import torch

# Fix seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
from models import run_baseline, run_bert

warnings.filterwarnings("ignore")

MIN_COUNT = 30
TRAIN_SAMPLE = 1500
VAL_SAMPLE = 400

if __name__ == "__main__":
    # Load data from parquet files
    train = pd.read_parquet("data/train-00000-of-00001-b21313e511aa601a.parquet")
    val = pd.read_parquet("data/val-00000-of-00001-66ce8665444026dc.parquet")
    
    # Optionally subsample for faster debugging or limited resources
    train = train.sample(TRAIN_SAMPLE, random_state=SEED)
    val = val.sample(VAL_SAMPLE, random_state=SEED)

    # Build text input for models by concatenating title and abstract
    train["full_text"] = train["title"].fillna('') + " " + train["abstract"].fillna('')
    val["full_text"] = val["title"].fillna('') + " " + val["abstract"].fillna('')

    # Multi-label binarization for all possible labels seen in both train and val
    all_labels = list(train["verified_uat_labels"]) + list(val["verified_uat_labels"])
    mlb = MultiLabelBinarizer()
    mlb.fit(all_labels)
    Y_train = mlb.transform(train["verified_uat_labels"])
    Y_val = mlb.transform(val["verified_uat_labels"])

    # Filter out rare labels
    label_counts = Y_train.sum(axis=0) + Y_val.sum(axis=0)
    labels_to_keep = np.where(label_counts >= MIN_COUNT)[0]
    Y_train = Y_train[:, labels_to_keep]
    Y_val = Y_val[:, labels_to_keep]
    label_names = mlb.classes_[labels_to_keep]
    print(f"Number of labels kept after filtering: {len(label_names)}")
    print("Choose the model:")
    print("1. TF-IDF + Logistic Regression")
    print("2. BERT (transformers)")
    choice = input("Your choice (1/2): ")

    if choice == "1":
        run_baseline(train, val, Y_train, Y_val, label_names)
    elif choice == "2":
        run_bert(train, val, Y_train, Y_val, label_names)
    else:
        print("Invalid choice!")