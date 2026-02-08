import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from imblearn.metrics import geometric_mean_score
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

MODEL_ID = "Qwen/Qwen3-8B"
TRAIN_PATH = "./prompts/llm_prompts715/train_prompts.csv"
VAL_PATH   = "./prompts/llm_prompts715/val_prompts.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2 
EPOCHS = 10
LR = 1e-4
HIDDEN_DIM = 512
DROPOUT = 0.4
USE_PCA = True
PCA_DIM = 512
OUTPUT_DIR = "./qwen3_mlp_results_715"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModel.from_pretrained(
    MODEL_ID, output_hidden_states=True, torch_dtype=torch.float16
).to(DEVICE).eval()

def encode_llama3(texts, batch_size=BATCH_SIZE, max_length=512):
    """Mean-pool last hidden states as sentence embedding."""
    embs = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state
        emb = out.mean(dim=1)
        embs.append(emb.cpu())
    return torch.cat(embs, dim=0)


train_df = pd.read_csv(TRAIN_PATH)
val_df   = pd.read_csv(VAL_PATH)

if not os.path.exists("train_emb.pt"):
    print(f"Encoding {len(train_df)} train prompts...")
    train_emb = encode_llama3(train_df["prompt"].tolist())
    print(f"Encoding {len(val_df)} val prompts...")
    val_emb   = encode_llama3(val_df["prompt"].tolist())
    torch.save(train_emb, "train_emb.pt")
    torch.save(val_emb, "val_emb.pt")
else:
    train_emb = torch.load("train_emb.pt")
    val_emb   = torch.load("val_emb.pt")

train_y = torch.tensor(train_df["label"].values, dtype=torch.long)
val_y   = torch.tensor(val_df["label"].values, dtype=torch.long)

scaler = StandardScaler()
train_X = scaler.fit_transform(train_emb)
val_X   = scaler.transform(val_emb)

if USE_PCA:
    print(f"Applying PCA to {PCA_DIM} dims...")
    pca = PCA(n_components=PCA_DIM)
    train_X = pca.fit_transform(train_X)
    val_X   = pca.transform(val_X)

train_X = torch.tensor(train_X, dtype=torch.float32)
val_X   = torch.tensor(val_X, dtype=torch.float32)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=3, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)

input_dim = train_X.shape[1]
model_mlp = MLPClassifier(input_dim, HIDDEN_DIM, num_classes=3, dropout=DROPOUT).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model_mlp.parameters(), lr=LR, weight_decay=1e-5)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_X, train_y), batch_size=32, shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(val_X, val_y), batch_size=64
)

best_f1 = 0
patience, patience_counter = 3, 0

for epoch in range(EPOCHS):
    model_mlp.train()
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model_mlp(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    # ---- Validation ----
    model_mlp.eval()
    preds, probs, labels = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE)
            logits = model_mlp(xb)
            prob = torch.softmax(logits, dim=1)
            preds.extend(prob.argmax(dim=1).cpu().tolist())
            probs.extend(prob.cpu().numpy())
            labels.extend(yb.tolist())

    # ---- Metrics ----
    macro_f1 = f1_score(labels, preds, average="macro")
    auc = roc_auc_score(label_binarize(labels, classes=[0,1,2]),
                        np.array(probs), average="macro", multi_class="ovr")
    gmean = geometric_mean_score(labels, preds, average="macro")

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")

    print(f"Epoch {epoch+1}/{EPOCHS} | "
        f"Loss: {avg_loss:.4f} | "
        f"F1: {macro_f1:.4f} | "
        f"AUC: {auc:.4f} | "
        f"G-Mean: {gmean:.4f} |"
        f"Acc: {acc:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f}"
        )

    # ---- Early Stopping ----
    if macro_f1 > best_f1:
        best_f1 = macro_f1
        patience_counter = 0
        torch.save(model_mlp.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

print(f"Best macro F1: {best_f1:.4f}")
