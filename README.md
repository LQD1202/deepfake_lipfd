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
conda create -n dfl python==3.10
conda activate dfl
```
### Install dependencies
```bash
pip install -r requirements.txt
```
## 2. Dataset Preparation

**Download Link: [AVLips v1.0](https://drive.google.com/file/d/1fEiUo22GBSnWD7nfEwDW86Eiza-pOEJm/view?usp=share_link)**

~~~
AVLips
├── 0_real
│   ├── 0.mp4
│    ...
├── 1_fake
│   ├── 0.mp4
│   └── ...
└── wav
    ├── 0_real
    │   ├── 0.wav
    │   └── ...
    └── 1_fake
        ├── 0.wav
        └── ...
~~~

Preprocess the dataset for training. 

~~~bash
python preprocess.py
~~~

Preprocessed AVLips dataset folder structure.

~~~bash
datasets
└── AVLips
    ├── 0_real
    │   ├── 0_0.png
    │   └── ...
    └── 1_fake
        ├── 0_0.png
        └── ...
~~~

The data sample is showed as follow, and **the fully processed dataset is approximately 60 GB.**

## :tada: Validation

- Download our [pertained weights](https://drive.google.com/file/d/1NPAcx0QS8N9v_9qUr-51jBaL9kGDT-cp/view?usp=share_link) and save it in to `checkpoints/ckpt.pth`. 

- Download [validation set](https://drive.google.com/file/d/1gZjzps5_rbr6CeBqBke8l2Gs8xXx_Ctb/view?usp=share_link) and extract it into `datasets/val`.

~~~bash
python validate.py --real_list_path ./datasets/val/0_real --fake_list_path ./datasets/val/1_fake --ckpt ./checkpoints/ckpt.pth
~~~



