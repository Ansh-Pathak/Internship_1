import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DATA_DIR = os.environ.get("DATA_DIR", "/kaggle/input/fashionmnist")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")
EPOCHS = int(os.environ.get("EPOCHS", 30))
BATCH_SIZE = 128
SEED = 42
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_df = pd.read_csv(os.path.join(DATA_DIR, "fashion-mnist_train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "fashion-mnist_test.csv"))
print("Train:", train_df.shape, "Test:", test_df.shape)
print(train_df["label"].value_counts().sort_index())

X_all = train_df.drop(columns="label").values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y_all = train_df["label"].values.astype(np.int64)
X_test = test_df.drop(columns="label").values.astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
y_test = test_df["label"].values.astype(np.int64)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.1, stratify=y_all, random_state=SEED)

MEAN, STD = float(X_train.mean()), float(X_train.std())
X_train = (X_train - MEAN) / STD
X_val = (X_val - MEAN) / STD
X_test = (X_test - MEAN) / STD
print("Split sizes:", len(X_train), len(X_val), len(X_test))

class FashionDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            if random.random() < 0.5:
                x = torch.flip(x, dims=[2])
            x = F.pad(x, (2, 2, 2, 2), value=float(x.min()))
            i = random.randint(0, 4)
            j = random.randint(0, 4)
            x = x[:, i:i + 28, j:j + 28]
        return x, self.y[idx]

def make_loaders(augment):
    train_loader = DataLoader(FashionDataset(X_train, y_train, augment),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(FashionDataset(X_val, y_val), batch_size=512, num_workers=2)
    test_loader = DataLoader(FashionDataset(X_test, y_test), batch_size=512, num_workers=2)
    return train_loader, val_loader, test_loader

fig, axes = plt.subplots(2, 8, figsize=(14, 4))
for ax, img, lab in zip(axes.ravel(), X_all[:16], y_all[:16]):
    ax.imshow(img[0], cmap="gray")
    ax.set_title(CLASS_NAMES[lab], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_images.png"), dpi=150)
plt.close()

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class FashionResNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU())
        self.stage1 = ResBlock(32, 32)
        self.stage2 = ResBlock(32, 64, stride=2)
        self.stage3 = ResBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print("MLP parameters:", count_params(MLP()))
print("ResNet parameters:", count_params(FashionResNet()))

def run_epoch(model, loader, criterion, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            n += len(yb)
    return total_loss / n, correct / n

def train_model(model, train_loader, val_loader, epochs, lr, weight_decay, patience=6):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state, wait = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scheduler)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping at epoch", epoch)
                break
    model.load_state_dict(best_state)
    return model, history

def predict(model, loader):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, _ in loader:
            p = F.softmax(model(xb.to(device)), dim=1).cpu()
            probs.append(p)
            preds.append(p.argmax(1))
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()

def evaluate(model, test_loader, name):
    preds, probs = predict(model, test_loader)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"\n{name}: test accuracy {acc:.4f} | macro F1 {macro_f1:.4f}")
    print(classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4))
    return {"accuracy": acc, "macro_f1": macro_f1, "preds": preds, "probs": probs}

def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_title(f"{name}: loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_title(f"{name}: accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"curves_{name}.png"), dpi=150)
    plt.close()

def plot_confusion(preds, name):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"confusion_{name}.png"), dpi=150)
    plt.close()

def plot_errors(preds, probs, name, k=12):
    wrong = np.where(preds != y_test)[0]
    conf = probs[wrong, preds[wrong]]
    chosen = wrong[np.argsort(-conf)[:k]]
    fig, axes = plt.subplots(2, k // 2, figsize=(14, 5))
    for ax, idx in zip(axes.ravel(), chosen):
        ax.imshow(X_test[idx, 0] * STD + MEAN, cmap="gray")
        ax.set_title(f"T: {CLASS_NAMES[y_test[idx]]}\nP: {CLASS_NAMES[preds[idx]]}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"errors_{name}.png"), dpi=150)
    plt.close()

experiments = [
    ("MLP_baseline", MLP, False, 1e-3, 1e-4),
    ("ResNet_no_augmentation", FashionResNet, False, 2e-3, 5e-4),
    ("ResNet_full", FashionResNet, True, 2e-3, 5e-4),
]

summary = {}
for name, model_cls, augment, lr, wd in experiments:
    print("\n" + "=" * 70)
    print("Experiment:", name)
    print("=" * 70)
    torch.manual_seed(SEED)
    train_loader, val_loader, test_loader = make_loaders(augment)
    model = model_cls()
    n_params = count_params(model)
    model, history = train_model(model, train_loader, val_loader, EPOCHS, lr, wd)
    result = evaluate(model, test_loader, name)
    plot_history(history, name)
    plot_confusion(result["preds"], name)
    plot_errors(result["preds"], result["probs"], name)
    summary[name] = {
        "parameters": n_params,
        "epochs_run": len(history["train_loss"]),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "best_val_loss": min(history["val_loss"]),
        "test_accuracy": result["accuracy"],
        "test_macro_f1": result["macro_f1"],
    }

with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(pd.DataFrame(summary).T.to_string())
torch.save(model.state_dict(), os.path.join(OUT_DIR, "fashion_resnet_best.pt"))

