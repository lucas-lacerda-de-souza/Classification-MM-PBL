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

│   ├── train/

│   ├── val/

│   └── test/

│       └── patch.png files


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

git clone https://github.com/lucas-lacerda-de-souza/Classification-MM-PBL.git
cd Classification-MM-PBL
________________________________________

**11. Quick Start Guide**

**11.1. Clone the repository**

git clone https://github.com/lucas-lacerda-de-souza/Classification-MM-PBL.git
cd Classification-MM-PBL

**11.2. Create and activate the environment**

conda env create -f environment.yml
conda activate mm-pbl-ai

**11.3. Run inference**

python inference.py --input_dir ./data/test/ --output_dir ./results/

**11.4. Generate Grad-CAM heatmaps**

python scripts/visualize_gradcam.py \
  --input_dir ./data/test/ \
  --output_dir ./gradcam/heatmaps/
________________________________________

**12. Compliance with TRIPOD-AI and CLAIM 2024 Guidelines**

This repository has been structured to meet the TRIPOD-AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis – 
AI extension) and CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging) requirements for transparent and reproducible AI in healthcare.

**Data Source and Splits**

Detailed in README.md → Dataset Organization and METHODS.md.
Data divided into 80% training, 10% validation, and 10% testing.
Two independent external validation cohorts used to assess generalizability.

**Model Architecture and Training**

Documented in /models and individual training scripts.
Includes optimizer (AdamW), learning rate, batch size, epochs, and loss functions.

**Performance Metrics**

Internal and external validation results summarized in /results
Cross-institutional evaluation demonstrates robustness to domain shifts.

**Interpretability and Explainability**

SHAP feature importance for XGBoost models and Grad-CAM heatmaps for CNNs included.
Code and examples available in /models and /data.

**Clinical and Biological Relevance**

Described in MODEL_CARD.md → Intended Use.
Designed to assist diagnostic workflows, not to replace expert evaluation.

**Limitations and Potential Biases**

Outlined in MODEL_CARD.
Includes dataset size, center-specific staining differences, and potential bias from single-institution data predominance.

**Ethical Considerations**

Discussed in MODEL_CARD.md → Ethical and Practical Considerations.
Model not intended for autonomous clinical use; human oversight required at all stages.

________________________________________

**13. Ethics**

This study was approved by the Ethics Committee of the Piracicaba Dental School, University of Campinas, Piracicaba, Brazil (protocol no. 67064422.9.1001.5418), 
and by the West of Scotland Research Ethics Service (20/WS/0017). The study was performed according to the clinical standards of the 1975 and 1983 Declaration of Helsinki. 
Written consent was not required as data was collected from surplus archived tissue. Data collected were fully anonymised.

________________________________________

**14. Data availability**

All the data derived from this study are included in the manuscript. We are unable to share the whole slide images and clinical data, due to restrictions in the 
ethics applications. However, we created synthetic slides to show the structure of the project.

________________________________________

**15. Code availability**

We have made the codes publicly available online, along with model weights (https://github.com/lucas-lacerda-de-souza/Classification-MM-PBL/tree/main). All code was written 
with Python Python 3.12.11, along with PyTorch 2.8.0. The full implementation of the model, including the code and documentation, has been deposited in the Zenodo repository 
and is publicly available (...). 

________________________________________
**16. Citation**

@article{delasouza2026classification,
  title={Machine Learning and Multimodal Deep Learning to Classify Plasmablastic Lymphoma from Multiple Myeloma},
  author={Souza, Lucas Lacerda de, Chen, Zhiyang […] Khurram, Syed Ali and Vargas, Pablo Agustin},
  journal={(oral oncology / 2026)},
  year={2026}
}
________________________________________
**17. License**

MIT License © 2025 Lucas Lacerda de Souza
