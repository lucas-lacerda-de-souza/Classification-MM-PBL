📘 Model Categories
1. Multimodal Deep Learning

These scripts implement a multimodal deep learning framework that performs late fusion between CNN-based image features and clinicopathological + nuclear morphometric embeddings for patch-level and patient-level classification of Multiple Myeloma (MM) and Plasmablastic Lymphoma (PBL) of the head and neck.

Supported CNN Architectures

AlexNet

ResNet50

DenseNet121

InceptionV3

CellViT

Scripts
Script	Level	Description
multimodal_alexnet_patch_level.py	Patch	AlexNet-based multimodal model for patch-level classification.
multimodal_alexnet_patient_level.py	Patient	Patient-level AlexNet inference using mean aggregation of patch probabilities.
multimodal_resnet50_patch_level.py	Patch	ResNet50 multimodal classifier integrating image patches and structured data.
multimodal_resnet50_patient_level.py	Patient	Aggregated patient-level inference based on ResNet50 outputs.
multimodal_densenet121_patch_level.py	Patch	DenseNet121-based multimodal fusion at patch level.
multimodal_densenet121_patient_level.py	Patient	DenseNet121 patient-level classification via probability aggregation.
multimodal_inceptionv3_patch_level.py	Patch	InceptionV3 multimodal model for patch-level prediction.
multimodal_inceptionv3_patient_level.py	Patient	Patient-level inference using InceptionV3 patch aggregation.
multimodal_cellvit_patch_level.py	Patch	CellViT-based multimodal fusion with transformer-based visual embeddings.
multimodal_cellvit_patient_level.py	Patient	High-performance patient-level CellViT inference.
All multimodal models include:

Late fusion (CNN + MLP embeddings)

Weighted cross-entropy to address class imbalance

ROC, AUC, precision–recall curves and confusion matrices

Patch-level and patient-level evaluation

Calibration analysis (ECE, Brier score)

Multi-GPU support (torch.nn.DataParallel)

Mixed-precision training (CUDA)

2. Image Segmentation
Script	Description
segmentation_unet++.py	U-Net++ with attention gates for tissue segmentation and ROI extraction prior to patch generation.

Purpose: removal of background and non-representative tissue before patch extraction.

3. Classical Machine Learning
Script	Description
xgboost_classification_cpc_mpa.R	XGBoost classifier trained on clinicopathological and nuclear morphometric variables.
xgboost_classification_gradcam.R	XGBoost model trained on Grad-CAM–derived nuclear features for explainability-driven validation.
Both scripts support:

Training / validation / test split

ROC and precision–recall evaluation

Confusion matrices

SHAP-based feature importance

CSV export of metrics to /results/

⚙️ Usage Examples
Train a multimodal ResNet50 model (patch-level)
python multimodal_resnet50_patch_level.py \
    --train_dir ./data/train \
    --val_dir ./data/val \
    --epochs 100 \
    --batch_size 64 \
    --output_dir ./results/resnet50_patch/

Run XGBoost Grad-CAM–guided analysis
python xgboost_classification_gradcam.R \
    --input ./supplementary_data/supplementary_table_4.xlsx \
    --output ./results/xgboost_gradcam/

📁 Output Structure
results/
│
├── resnet50_patch/
│   ├── model.pth
│   ├── metrics.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
│
├── cellvit_patient/
│   ├── model.pth
│   ├── calibration_curve.png
│   ├── roc_curve.png
│   └── performance_metrics.csv
│
├── unetpp_segmentation/
│   ├── masks/
│   └── training_curves.png
│
└── xgboost_gradcam/
    ├── feature_importance.png
    ├── shap_summary.png
    └── performance_metrics.csv

⚙️ Model Configuration and Training Details
1. Traditional Machine Learning (XGBoost)

Objective: Binary logistic regression (MM vs PBL)

Boosting rounds: 100

Learning rate: 0.1

Maximum tree depth: 6

Train/test split: 70% / 30%

Evaluation metrics: Accuracy, AUC, F1-score, Precision, Recall

Feature interpretability: SHAP (features retained if SHAP > 1.0)

2. Nuclear Segmentation (U-Net++)

Architecture: U-Net++ with attention gates

Framework: PyTorch

Input size: 256 × 256 pixels

Training dataset: NuInsSeg (~30,000 annotated nuclei)

Data split: 80% training / 10% validation / 10% test

Optimizer: Adam (learning rate = 1 × 10⁻⁴)

Loss function: Binary Cross-Entropy with Logits

Batch size: 4

Epochs: 50

Precision mode: Mixed precision (CUDA)

3. Patch Generation and Pre-processing

Patch size: 299 × 299 pixels

Patch overlap: 20%

Color normalization: Macenko method

Data augmentation:

Rotations (90°, 180°, 270°)

Gaussian blur

Zoom (+20%)

Dataset (MM vs PBL)

Total cases: 62 (31 MM, 31 PBL)

Internal split (patient-level):

Training (80%): 50 cases

MM: 70,934 patches

PBL: 69,389 patches

After augmentation:

MM: 425,604 patches

PBL: 416,334 patches

External validation:

Two independent cohorts from unseen centers

4. Multimodal Deep Learning (CNN + MLP Fusion)

Backbones: AlexNet, ResNet50, DenseNet121, InceptionV3, CellViT (ImageNet pretrained)

Fusion strategy: Late fusion (concatenation of CNN and MLP embeddings)

MLP branch: Fully connected layers with ReLU activation and dropout

Optimizer: AdamW

Learning rate: 1 × 10⁻⁴

Weight decay: 1 × 10⁻⁴

Loss function: Weighted Cross-Entropy

Batch size: 64

Computation: Mixed precision (CUDA)

Patient-level aggregation: Mean of patch-level probabilities

Evaluation:

Accuracy, Precision, Recall, F1-score

ROC AUC (DeLong + bootstrap)

Cohen’s kappa

Calibration (ECE, Brier score)

🧬 Reproducibility Notes

Python: 3.12.11

PyTorch: 2.8.0 + CUDA 12.8

GPU: NVIDIA RTX 3090 (24 GB)

OS: Ubuntu 20.04 LTS

RAM: 125 GB

Deterministic behavior ensured via fixed random seeds (torch, numpy, random)
