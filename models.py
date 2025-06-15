import random
import numpy as np
import torch

# Fix seed for reproducibility (optionnel ici si déjà fait dans main.py, mais conseillé si tu veux pouvoir exécuter models.py tout seul)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, classification_report
def run_baseline(train, val, Y_train, Y_val, label_names, max_features=1000, threshold=0.15):
    # Vectorize text data using TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train = vectorizer.fit_transform(train["full_text"])
    X_val = vectorizer.transform(val["full_text"])

    # Train a logistic regression classifier in a one-vs-rest multilabel setting
    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=300, solver='saga', n_jobs=-1, C=0.5)
    )
    clf.fit(X_train, Y_train)

    # Predict probabilities, then apply a threshold to get binary predictions
    Y_pred_proba = clf.predict_proba(X_val)
    Y_pred = (Y_pred_proba > threshold).astype(int)

    # Show metrics only for labels present in the validation set
    present_in_val = (Y_val.sum(axis=0) > 0)
    print("F1-score (micro):", f1_score(Y_val[:, present_in_val], Y_pred[:, present_in_val], average='micro'))
    print(classification_report(
        Y_val[:, present_in_val],
        Y_pred[:, present_in_val],
        target_names=label_names[present_in_val]
    ))

    # Print prediction examples for a few articles
    for i in range(3):
        print(f"\nARTICLE {i+1}")
        print("TEXT:", val.iloc[i]["full_text"][:300], "...")
        print("TRUE:", [label_names[j] for j, v in enumerate(Y_val[i]) if v])
        print("PRED:", [label_names[j] for j, v in enumerate(Y_pred[i]) if v])

def run_bert(train, val, Y_train, Y_val, label_names, max_length=128, batch_size=8, epochs=3, threshold=0.05):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import BertTokenizerFast, BertForSequenceClassification
    from torch.optim import AdamW
    from tqdm import tqdm

    print("Running BERT (multi-label classification)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = Y_train.shape[1]
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizerFast.from_pretrained(model_name)

    # Custom Dataset class to tokenize texts and provide labels
    class TextDataset(Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=max_length)
            self.labels = labels
        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx]).float()
            return item
        def __len__(self):
            return len(self.labels)

    train_dataset = TextDataset(train["full_text"], Y_train)
    val_dataset = TextDataset(val["full_text"], Y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load BERT model for sequence classification with multilabel output
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, problem_type="multi_label_classification"
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    # Evaluation on validation set
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))
            preds = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_labels.append(labels)
    Y_pred = np.vstack(all_preds)
    Y_true = np.vstack(all_labels)

    present_in_val = (Y_true.sum(axis=0) > 0)
    print("F1-score (micro):", f1_score(Y_true[:, present_in_val], Y_pred[:, present_in_val], average='micro'))
    print(classification_report(
        Y_true[:, present_in_val],
        Y_pred[:, present_in_val],
        target_names=label_names[present_in_val]
    ))

    # Show example predictions for a few articles
    for i in range(3):
        print(f"\nARTICLE {i+1}")
        print("TEXT:", val.iloc[i]["full_text"][:300], "...")
        print("TRUE:", [label_names[j] for j, v in enumerate(Y_true[i]) if v])
        print("PRED:", [label_names[j] for j, v in enumerate(Y_pred[i]) if v])