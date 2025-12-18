**Machine Learning and Multimodal Deep Learning to Classify Plasmablastic Lymphoma from Multiple Myeloma** 

Author: Lucas Lacerda de Souza

Year: 2025
________________________________________
**1. Project Overview**

This study proposes an explainable multimodal deep learning framework to distinguish multiple myeloma from plasmablastic lymphoma by integrating three complementary data streams: 
(I) histopathological image patches analysed using established convolutional neural network architectures (AlexNet, ResNet50, DenseNet121, InceptionV3) and a transformer-based model (CellViT); 
(II) structured clinicopathological variables; and (III) quantitative nuclear morphometric features modelled using gradient-boosting classifiers. Model interpretability is incorporated through 
Grad-CAM visualisation for image-based predictions and SHAP-based feature attribution, aiming to support diagnostic decision-making in challenging plasma-cell neoplasms.
________________________________________
**2. Pipeline**

<img width="2000" height="1155" alt="Figure 2" src="https://github.com/user-attachments/assets/b98c33fa-c12d-4f0b-bd9e-1ac8d21db3f1" />

________________________________________

**3. Environment and Hardware**

All experiments were performed using the following configuration:

**Operating System:** Ubuntu 20.04.1 LTS

**Python Version:** 3.12.11

**PyTorch Version:** 2.8.0 (CUDA 12.8)

**CPU:** Intel Xeon W-2295 (18 cores / 36 threads)

**RAM:** 125 GB

**GPUs:** 3 × NVIDIA GeForce RTX 3090 (24 GB each)
________________________________________
**4. Environment Files**

**Channels:**

  • pytorch
  
  • nvidia
  
  • defaults
  
**Dependencies:**

  • python=3.12.11
  
  • pytorch=2.8.0
  
  • torchvision=0.19.0
  
  • torchaudio=2.8.0
  
  • cudatoolkit=12.8
  
  • numpy=1.26.4
  
  • pandas=2.2.3
  
  • scikit-learn=1.5.2
  
  • matplotlib=3.9.2
  
  • seaborn=0.13.2
  
  • pillow=10.4.0
  
  • tqdm=4.66.5
  
  • openpyxl=3.1.5
________________________________________
**5. Model Architectures**

  • XGBoost with SHAP explainability

  • AlexNet with Multilayer Perceptron (MLP) fusion

  • ResNet50 with Multilayer Perceptron (MLP) fusion

  • DenseNet121 with Multilayer Perceptron (MLP) fusion

  • InceptionV3 with Multilayer Perceptron (MLP) fusion

  • CellViT with Multilayer Perceptron (MLP) fusion

  • Grad-CAM (Gradient-weighted Class Activation Mapping)
________________________________________
**6. Features Used**

• Patches (H&E)
   
•	Morphometric features (nucleus-based)

•	Clinicopathologic features (age, sex, location)
________________________________________
**7. Evaluation Metrics**
   
•	XGBoost + SHAP (clinicopathological and morphometric data)
Classification performance was evaluated using accuracy, area under the receiver operating characteristic curve (AUC), F1-score, precision, and recall.
Model interpretability was assessed using SHAP values, with summary and decision plots used to quantify feature contribution.

•	AlexNet (multimodal deep learning)
Performance was assessed using loss, accuracy, precision, recall, confusion matrix components (true positives, false negatives, false positives, true negatives), F1-score, specificity, receiver operating characteristic area under the curve (ROC AUC), and Cohen’s kappa coefficient.

•	ResNet50 (multimodal deep learning)
Evaluation included loss, accuracy, precision, recall, confusion matrix components (TP, FN, FP, TN), F1-score, specificity, ROC AUC, and Cohen’s kappa coefficient.

•	DenseNet121 (multimodal deep learning)
Model performance was measured using loss, accuracy, precision, recall, confusion matrix components (TP, FN, FP, TN), F1-score, specificity, ROC AUC, and Cohen’s kappa coefficient.

•	InceptionV3 (multimodal deep learning)
Evaluation metrics comprised loss, accuracy, precision, recall, confusion matrix components (TP, FN, FP, TN), F1-score, specificity, ROC AUC, and Cohen’s kappa coefficient.

•	CellViT (multimodal deep learning)
Performance assessment included loss, accuracy, precision, recall, confusion matrix components (TP, FN, FP, TN), F1-score, specificity, ROC AUC, and Cohen’s kappa coefficient, with additional emphasis on probability calibration using the Expected Calibration Error (ECE) and Brier score.

•	Patient-level evaluation (all multimodal models)
Patient-level discrimination was evaluated using ROC AUC with 95% confidence intervals, estimated via DeLong’s method and bootstrap resampling.
Sensitivity, specificity, and accuracy were assessed across multiple probability thresholds, with the optimal cut-off defined by the Youden index.

•	Grad-CAM–guided XGBoost classification (explainability-driven analysis)
Classification performance based on nuclear features extracted from Grad-CAM–highlighted regions was evaluated using accuracy, AUC, F1-score, precision, and recall.

________________________________________
**8. Repository Structure**
   
## 📂 Repository Structure

INFERENCE.py — Inference Script Example

LICENSE.txt — Project license

MODEL_CARD.txt — Description of the essential information of the study 

README.md — Documentation and usage instructions

REQUIREMENTS.txt — Dependencies


data/

patches/

├── gradcam/

│   ├── heatmaps/

│   │   └── heatmap.png files

│   ├── patches/

│   │   └── patch.png files

│   └── wsi_heatmaps/

│       └── wsi.png files

│

├── masks/

│   ├── train/

│   ├── val/

│   └── test/

│       └── mask.png files

│

└── patches/

    ├── train/
    
    ├── val/
    
    └── test/
    
        └── patch.png files


models/

├── AlexNet/

│   ├── multimodal_alexnet_patch_level.py

│   └── multimodal_alexnet_patient_level.py

│

├── ResNet50/

│   ├── multimodal_resnet50_patch_level.py

│   └── multimodal_resnet50_patient_level.py

│

├── DenseNet121/

│   ├── multimodal_densenet121_patch_level.py

│   └── multimodal_densenet121_patient_level.py

│

├── InceptionV3/

│   ├── multimodal_inceptionV3_patch_level.py

│   └── multimodal_inceptionV3_patient_level.py

│

├── CellViT/

│   ├── multimodal_cellvit_patch_level.py

│   └── multimodal_cellvit_patient_level.py

│

└── XGBoost/

│   ├── xgboost_classification_cpc_mpa.R

│   └── xgboost_classification_gradcam.R
    
results/

└── metrics/

________________________________________

**9. Run models and reproduce tables**

<img width="2000" height="501" alt="image" src="https://github.com/user-attachments/assets/17771059-2f3e-40c5-ac3e-512e19a81e9f" />

________________________________________

**10. Installation**

git clone https://github.com/lucas-lacerda-de-souza/Classification-DLBCL-ENKTCL-NT.git
cd Classification-DLBCL-ENKTCL-NT

________________________________________





  
