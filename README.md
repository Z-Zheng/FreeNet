[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Travis-CI](https://travis-ci.org/LiyuanLucasLiu/RAdam.svg?branch=master)](https://travis-ci.org/LiyuanLucasLiu/RAdam)



<h1 align="center">FPGA & FreeNet</h1>
<h5 align="center">Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification</h5>

<h5 align="right">by <a href="http://zhuozheng.top/">Zhuo Zheng</a>, <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>, Ailong Ma and <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a></h5>

<div align="center">
  <img src="https://github.com/Z-Zheng/images_repo/raw/master/fpga.png"><br><br>
</div>

This is an official implementation of FPGA framework and FreeNet in our TGRS 2020 paper ["FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification"](https://ieeexplore.ieee.org/document/9007624).

We hope the FPGA framework can become a stronger and cleaner baseline for hyperspectral image classification research in the future.

## News
1. 2020/05/28, We release the code of FreeNet and FPGA framework.


## Features
1. Patch-free training and inference
2. Fully end-to-end (w/o preprocess technologies, such as dimension reduction)


## Citation
If you use FPGA framework or FreeNet in your research, please cite the following paper:
```text
@article{zheng2020fpga,
  title={FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification},
  author={Zheng, Zhuo and Zhong, Yanfei and Ma, Ailong and Zhang, Liangpei},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2020},
  publisher={IEEE},
  note={doi: {10.1109/TGRS.2020.2967821}}
}
```
 

## Getting Started
### 1. Install SimpleCV

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```
### 2. Prepare datasets

It is recommended to symlink the dataset root to `$FreeNet`.

The project should be organized as:
```text
FreeNet
├── configs     // configure files
├── data        // dataset and dataloader class
├── module      // network arch.
├── scripts 
├── pavia       // data 1
│   ├── PaviaU.mat
│   ├── PaviaU_gt.mat
├── salinas     // data 2
│   ├── Salinas_corrected.mat
│   ├── Salinas_gt.mat
├── GRSS2013    // data 3
│   ├── 2013_IEEE_GRSS_DF_Contest_CASI.tif
│   ├── train_roi.tif
│   ├── val_roi.tif
```

### 3. run experiments

#### 1. PaviaU
```bash
bash scripts/freenet_1_0_pavia.sh
```

#### 2. Salinas
```bash
bash scripts/freenet_1_0_salinas.sh
```

#### 3. GRSS2013
```bash
bash scripts/freenet_1_0_grss.sh
```

### License
This source code is released under [GPLv3](http://www.gnu.org/licenses/) license.

For commercial use, please contact Prof. Zhong (zhongyanfei@whu.edu.cn).