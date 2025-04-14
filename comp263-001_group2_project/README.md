# Brain Tumor Detection using Deep Learning
## COMP263-001 Group 2 Project

## Team Members
- Sandeep Neupane - Email: sneupa23@my.centennialcollege.ca
- Jessica Marie Hernandez - Email: jherna80@my.centennialcollege.ca
- Yi-Chen Hsu - Email: yhsu15@my.centennialcollege.ca
- Diego Armando Sarmiento Ahumada - Email: dsarmie7@my.centennialcollege.ca
- Tsang Kwong Ngan - Email: tngan@my.centennialcollege.ca
- Arcan Caglayan - Email: acaglay1@my.centennialcollege.ca

## Project Overview
This project implements and compares various deep learning approaches for brain tumor detection from MRI images. The implemented models include:

1. **Supervised Learning**:
   - VGG16-based CNN classification model

2. **Unsupervised Learning**:
   - Convolutional Autoencoder for anomaly detection
   - Variational Autoencoder (VAE)

3. **State-of-the-Art Models**:
   - EfficientNet B0
   - Vision Transformer (ViT)

## Directory Structure
```
comp263-001_group2_project/
├── scripts/
│   ├── brain_mri_preprocessor.py    # Data preprocessing utilities
│   ├── supervised_learning-VGG_CNN.py # VGG16 model implementation
│   ├── unsupervised_autoencoder.py  # Autoencoder implementation
│   ├── unsupervised_VAE.py         # VAE implementation
│   ├── sota_EfficientNet.py        # EfficientNet implementation
│   ├── sota_VIT.py                 # Vision Transformer implementation
│   └── final_comp.py               # Results comparison
├── reports/
│   ├── comp263_001_group2_analysis_report.pdf  # Detailed analysis report
│   └── comp263-001_group2_presentation.pptx    # Project presentation
├── results/                        # Created when running code
│   ├── effnet/                     # EfficientNet detailed results
│   ├── supervised/                 # VGG16 model outputs
│   ├── unsupervised/               # Autoencoder and VAE outputs
│   └── sota/                       # State-of-the-art model outputs
└── README.md                       # This file
```

## Setup and Requirements

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Dataset
The project uses the Brain MRI Images for Brain Tumor Detection dataset from Kaggle. The code includes functionality to automatically download the dataset using the Kaggle API.

#### Setting up Kaggle API
1. Create a Kaggle account
2. Generate API token from your account settings
3. Place `kaggle.json` in:
   - Windows: `C:\Users\<YourUserName>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`

## Usage

### Data Preprocessing
```bash
python scripts/brain_mri_preprocessor.py
```

### Running Models

1. **Supervised Learning (VGG16)**:
```bash
python scripts/supervised_learning-VGG_CNN.py
```

2. **Unsupervised Learning**:
```bash
# Autoencoder
python scripts/unsupervised_autoencoder.py

# VAE
python scripts/unsupervised_VAE.py
```

### Comparing Results
```bash
python scripts/final_comp.py
```

## Results

### Model Performance Comparison

| Criteria | VGG | Autoencoder | VAE | Vision Transformer | EfficientNet |
|----------|-----|-------------|-----|-------------------|-------------|
| Model Architecture | VGG16 (CNN) | Denoising Autoencoder | Variational Autoencoder | ViT (Transformer) | EfficientNetB0 |
| Training Method | Supervised | Unsupervised | Unsupervised | Supervised | Transfer Learning |
| Accuracy | 85.71% | 77.00% | 73.00% | 82.00% | 88.89% |
| Recall | 69.23% | 53.85% | 46.15% | 61.54% | 76.92% |
| Precision | 90.00% | 53.85% | 46.15% | 80.00% | 100.00% |
| F1-Score | 78.26% | 62.00% | 51.00% | 69.57% | 87.00% |
| False Negatives | 4 | 6 | 7 | 5 | 3 |
| False Positives | 1 | 0 | 0 | 2 | 0 |

*Note: Metrics are based on the official analysis report (comp263_001_group2_analysis_report.pdf)

### Key Findings

1. **Model Performance**:
   - EfficientNet achieved the best overall performance with 88.89% accuracy and perfect precision
   - VGG16 showed strong performance with 85.71% accuracy and 90% precision
   - Vision Transformer demonstrated good results with 82% accuracy
   - Unsupervised models (Autoencoder and VAE) showed promising results for anomaly detection

2. **Strengths and Weaknesses**:
   - EfficientNet: Highest accuracy and perfect precision, but more complex architecture
   - VGG16: High precision and simpler to train, but misses some tumor cases
   - Vision Transformer: Good balance of precision and recall, but more complex to train
   - Autoencoder: Strong "no tumor" detection with no false positives, but moderate recall
   - VAE: No false positives on non-tumor cases, but lowest recall among all models

3. **Practical Applications**:
   - For clinical settings requiring high precision: EfficientNet or VGG16
   - For screening applications where recall is critical: EfficientNet
   - For unsupervised anomaly detection: Autoencoder

### Detailed Results

Detailed results, including confusion matrices, ROC curves, and training metrics, can be found in the analysis report in the `reports` directory.

## Troubleshooting
- For dataset download issues: Download manually from Kaggle
- For memory issues: Reduce batch size in model parameters
- For GPU issues: Set `TF_MEMORY_ALLOCATION='0.7'`

## License
MIT License

Copyright (c) 2023 COMP263-001 Group 2

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.