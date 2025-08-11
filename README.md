# DeepFake Lip-Sync Detection (deepfake_lipfd)

This repository contains code for **Lightweight Audio-Visual DeepFake Lip-Sync Detection** using MobileNetV3, ShuffleNetV2, and audio-visual feature fusion.  
It supports **training, validation, and inference** for real-time DeepFake detection.


---
## 1. Installation

### Clone the repository
```bash
git clone https://github.com/LQD1202/deepfake_lipfd.git
cd deepfake_lipfd
```
### Create environment
```bash
conda create -n dfl python=3.10
conda activate dfl
```
### Install dependencies
```bash
pip install -r requirements.txt
```
## 2. Dataset Preparation
AVLips_preproccess/
│
├── 0_real/           # Real videos (.mp4)
├── 1_fake/           # Fake videos (.mp4)
├── wav/
│   ├── 0_real/       # Real audio (.wav)
│   └── 1_fake/       # Fake audio (.wav)



