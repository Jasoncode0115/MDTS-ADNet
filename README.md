# MDTS-ADNet
## Offcial implementation of "Intelligent traffic accident detection system in complex dynamic scenarios based on the dual-stream spatiotemporal-fusion model".

![pipeline](./MDTS-ADNet_files/model.png)
## 1. Dependencies
```
python==3.6.13
torch==1.10.2+cu113
torchvision==0.11.3+cu113
torchaudio==0.10.2+cu113
torch-geometric==2.0.3
numpy==1.19.5
pandas==1.1.5
opencv==4.5.3
pillow==6.2.2
matplotlib==3.3.4
scikit-learn==0.24.2
scipy==1.5.3
tqdm==4.63.0
yacs==0.1.8
pyyaml==6.0.1
requests==2.27.1
protobuf==3.19.6
```
## 2. Usage
### 2.1 Data preparation

![pipeline](./MDTS-ADNet_files/dataset.png)

We created a large traffic accident dataset called 4M-TAD containing frame-level labels standardized for unsupervised traffic accident detection.

Due to confidentiality requirements, only a part of the 4M-TAD dataset is being disclosed at this time:

[4M-TAD_Part_1](https://drive.google.com/file/d/1WXcdSDeiVRNw4gcVnvqtQmVdFDpCtecN/view?usp=sharing)

If you need access to the entire dataset for research purposes, please contact me via email at huxiaolong01155@gmail.com.

### 2.2 Train
To train the model , run:
```python
$ python Trian.py
```
### 2.3 Evaluation
To evaluate the model , run:
```python
$ python Evaluate.py
```

## 3. Results
AUC is the core indicator.

|     Model      | 4M-TAD | AI City Challenge 2021 | 
| :------------: | :-------: | :---------: | 
|    MDTS-ADNet    |   84.33%   |    82.14%    | 

## Acknowledgment
We thank cvlab-yonsei for the PyTorch implementation of the [MNAD](https://github.com/cvlab-yonsei/MNAD).

## Citation
If you find this repo useful, please consider citing:
```
@inproceedings{liu2025MDTS-ADNet,
title = {Intelligent traffic accident detection system in complex dynamic scenariosbased on the dual-stream spatiotemporal-fusion model},
author = {Huilin Liu, Xiaolong Hu, Guanghan Sun, Wenkang Zhang, Jialei Zhan, Haobo Fang
Yan Li, Wangi Ma},
booktitle={Engineering Applications of Artificial Intelligence},
year = {2025}
}
```
