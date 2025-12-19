"""
Multimodal CellViT Classifier for MM and PBL
-------------------------------------------------------------------------------------------
Author: Lucas Lacerda de Souza

Description: 
This script implements a multimodal deep learning pipeline that integrates histopathological image patches, clinicopathologic and nuclear morphometric features 
to classify plasma cell lesions into Multiple Myeloma (MM – Class 1) and Plasmablastic Lymphoma (PBL – Class 2). 

The model is based on a CellViT (Vision Transformer) backbone for image embeddings, combined with a fully-connected network for clinical and nuclear 
morphometric data.
    
Dependencies: 
torch>=2.1.0 
torchvision>=0.16.0 
pandas>=2.0.0 
numpy>=1.24.0 
matplotlib>=3.8.0 
seaborn>=0.13.0 
scikit-learn>=1.3.0 
pillow>=10.0.0 
tqdm>=4.66.0 
openpyxl>=3.1.0 

"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    roc_curve, cohen_kappa_score, brier_score_loss
)
from sklearn.calibration import calibration_curve
from scipy.stats import bootstrap
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import timm   # 🔴 NEW


# ===============================================================
# Dataset Definition
# ===============================================================
class MultimodalDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        if "Classe" not in dataframe.columns or "CaseID" not in dataframe.columns:
            raise ValueError("Excel file must contain 'Classe' and 'CaseID' columns.")

        self.image_dir = image_dir
        self.transform = transform
        self.items = []
        self.clinical_cols = [c for c in dataframe.columns if c not in ["Classe", "CaseID"]]

        for _, row in dataframe.iterrows():
            class_dir = os.path.join(image_dir, str(row["Classe"]))
            case_dir = os.path.join(class_dir, str(row["CaseID"]))
            if not os.path.isdir(case_dir):
                continue

            clinical_vec = [float(row[c]) for c in self.clinical_cols]

            for root, _, files in os.walk(case_dir):
                for f in files:
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.items.append({
                            "patch_path": os.path.join(root, f),
                            "clinical": clinical_vec,
                            "label": int(row["Classe"]),
                            "case_id": str(row["CaseID"])
                        })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sample = self.items[idx]
        image = Image.open(sample["patch_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        clinical = torch.tensor(sample["clinical"], dtype=torch.float32)
        label = torch.tensor(sample["label"], dtype=torch.long)

        return image, clinical, label, sample["case_id"]


# ===============================================================
# Multimodal CellViT Model 
# ===============================================================
class MultimodalCellViT(nn.Module):
    def __init__(self, clinical_input_dim, num_classes=2):
        super().__init__()

        # 🔴 CellViT backbone (ViT-B/16)
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0  # remove classifier
        )

        self.feature_dim = 768  # ViT-B/16 embedding dim

        self.dropout = nn.Dropout(0.5)

        # Clinical branch
        self.clinical_net = nn.Sequential(
            nn.Linear(clinical_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # Multimodal classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(512, num_classes)
        )

    def forward(self, image, clinical_data):
        x = self.backbone(image)          # CLS token embedding
        x = self.dropout(x)
        clinical_features = self.clinical_net(clinical_data)
        combined = torch.cat((x, clinical_features), dim=1)
        return self.classifier(combined)


# ===============================================================
# Patient-Level Evaluation 
# ===============================================================
def patient_level_report(df, output_dir=None, n_boot=1000, alpha=0.95, ece_bins=10):
    df_agg = df.groupby("patient_id").agg({"y_true": "max", "y_prob": "mean"}).reset_index()
    y_true = df_agg["y_true"].values
    y_prob = df_agg["y_prob"].values

    auc = roc_auc_score(y_true, y_prob)
    auc_ci = bootstrap(
        (y_true, y_prob),
        lambda yt, yp: roc_auc_score(yt, yp),
        n_resamples=n_boot,
        confidence_level=alpha,
        paired=True,
        random_state=42
    )

    auc_low, auc_high = auc_ci.confidence_interval

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=ece_bins)
    ece = np.mean(np.abs(frac_pos - mean_pred))
    brier = brier_score_loss(y_true, y_prob)

    thresholds = np.linspace(0, 1, 101)
    sens, spec = [], []

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens.append(tp / (tp + fn + 1e-9))
        spec.append(tn / (tn + fp + 1e-9))

    best_t = thresholds[np.argmax(np.array(sens) + np.array(spec) - 1)]
    final_preds = (y_prob >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, final_preds).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.legend()
        plt.savefig(f"{output_dir}/roc.png")
        plt.close()

    return {
        "auc": auc,
        "auc_ci_low": auc_low,
        "auc_ci_high": auc_high,
        "ece": ece,
        "brier": brier,
        "accuracy": acc
    }


# ===============================================================
# Main
# ===============================================================
def main():
    train_dir = "data/train"
    test_dir = "data/test"
    results_dir = "results/cellvit_multimodal"
    os.makedirs(results_dir, exist_ok=True)

    train_df = pd.read_excel(os.path.join(train_dir, "clinical_data_train.xlsx"))
    test_df = pd.read_excel(os.path.join(test_dir, "clinical_data_test.xlsx"))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)  # ViT-friendly
    ])

    train_dataset = MultimodalDataset(train_df, train_dir, transform)
    test_dataset = MultimodalDataset(test_df, test_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=8)

    model = MultimodalCellViT(len(train_dataset.clinical_cols))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training
    model.train()
    for images, clinical, labels, _ in tqdm(train_loader, desc="Training"):
        images, clinical, labels = images.to(device), clinical.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images, clinical), labels)
        loss.backward()
        optimizer.step()

    # Testing
    model.eval()
    y_true, y_prob, ids = [], [], []

    with torch.no_grad():
        for images, clinical, labels, case_ids in test_loader:
            images, clinical = images.to(device), clinical.to(device)
            probs = torch.softmax(model(images, clinical), dim=1)
            y_true.extend(labels.numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
            ids.extend(case_ids)

    df = pd.DataFrame({"patient_id": ids, "y_true": y_true, "y_prob": y_prob})
    metrics = patient_level_report(df, output_dir=results_dir)

    print(f"\nPatient-level AUC: {metrics['auc']:.3f}")


if __name__ == "__main__":
    main()
