# PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.13401)

---
## 📢 Latest Updates
- **20 Sep 2024**: The arXiv version is released [here](https://arxiv.org/abs/2409.13401).
 The code will be released soon. 
---



## 🎨 Overview

![PDF Page](assets/overview.jpg)
Segment Anything Model (SAM) is an advanced foundational model for image segmentation, widely applied to remote sensing images (RSIs). Due to the domain gap between RSIs and natural images, traditional methods typically use SAM as a source pre-trained model and fine-tune it with fully supervised masks. Unlike these methods, our work focuses on fine-tuning SAM using more convenient and challenging point annotations. Leveraging SAM’s zero-shot capability, we adopt a self-training framework that iteratively generates pseudo-labels. However, noisy labels in pseudo-labels can cause error accumulation. To address this, we introduce Prototype-based Regularization, where target prototypes are extracted from the dataset and matched to predicted prototypes using the Hungarian algorithm to guide learning in the correct direction. Additionally, RSIs have complex backgrounds and densely packed objects, making it possible for point prompts to mistakenly group multiple objects as one. To resolve this, we propose a Negative Prompt Calibration method, based on the non-overlapping nature of instance masks, where overlapping masks are used as negative signals to refine segmentation. Combining these techniques, we present a novel pointly-supervised segment anything model, PointSAM. We conduct experiments on three RSI datasets, including WHU, HRSID, and NWPU VHR-10, showing that our method significantly outperforms direct testing with SAM, SAM2, and other comparison methods. Additionally, PointSAM can act as a point-to-box converter for oriented object detection, achieving promising results and indicating its potential for other point-supervised tasks.

## 🎮 Getting Started
### 1.Install Environment
```bash
conda create --name pointsam python=3.10
conda activate pointsam

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
git clone https://github.com/Lans1ng/PointSAM.git
cd PointSAM
pip install -r requirements.txt
```

### 2.Prepare Dataset 

#### WHU Building Dataset

- Dataset download address: [WHU Building Dataset](https://aistudio.baidu.com/datasetdetail/56502)。

- For converting semantic label to instance label, you can refer to corresponding [conversion script](https://github.com/KyanChen/RSPrompter/blob/release/tools/rsprompter/whu2coco.py).

#### HRSID Dataset

- Dataset download address: [HRSID Dataset](https://github.com/chaozhong2010/HRSID).

#### NWPU VHR-10 Dataset

- Dataset download address: [NWPU VHR-10 Dataset](https://aistudio.baidu.com/datasetdetail/52812).

- Instance label download address: [NWPU VHR-10 Instance Label](https://github.com/chaozhong2010/VHR-10_dataset_coco).

For convenience, we have included all the JSON annotations in this repo, and you only need to download the corresponding images. Specifically, organize the dataset as follows:

```
data 
├── WHU
│    ├── annotations
│    │   ├── WHU_building_train.json
│    │   ├── WHU_building_test.json
│    │   └── WHU_building_val.json
│    └── images
│        ├── train
│        │    ├── image
│        │    └── label
│        ├── val
│        │    ├── image
│        │    └── label
│        └── test
│             ├── image
│             └── label
├── HRSID
│    ├── Annotations
│    │   ├── all
│    │   ├── inshore
│    │   │      ├── inshore_test.json
│    │   │      └── inshore_train.json       
│    │   └── offshore
│    └── Images
└── NWPU
     ├── Annotations
     │   ├── NWPU_instnaces_train.json
     │   └── NWPU_instnaces_val.json
     └── Images

```
### 3.Training
For convenience, the `scripts` folder contains instructions for **Supervised Training**, **Self-Training**, and **PointSAM** on the NWPU VHR-10, WHU, and HRSID datasets.

Here is an example of using PointSAM to train on the WHU dataset.
```bash
bash scripts/train_whu_pointsam.sh
```

## 💡 Acknowledgement

- [wesam](https://github.com/zhang-haojie/wesam)
- [OWOD](https://github.com/JosephKJ/OWOD)
- [RSPrompter](https://github.com/KyanChen/RSPrompter)


## 🖊️ Citation

If you find this project useful in your research, please consider starring ⭐ and citing 📚:

```BibTeX
@article{liu2024pointsam,
  title={PointSAM: Pointly-Supervised Segment Anything Model for Remote Sensing Images},
  author={Liu, Nanqing and Xu, Xun and Su, Yongyi and Zhang, Haojie and Li, Heng-Chao},
  journal={arXiv preprint arXiv:2409.13401},
  year={2024}
}
```
