# FreeNet
By [Zhuo Zheng](http://zhuozheng.top/), [Yanfei Zhong](http://rsidea.whu.edu.cn/), Ailong Ma and [Liangpei Zhang](http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html)

<div align="center">
  <img src="https://github.com/Z-Zheng/images_repo/raw/master/fpga.png"><br><br>
</div>

This is an official implementation of FreeNet in our TGRS 2020 paper ["FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification"](https://ieeexplore.ieee.org/document/9007624).

We hope the FreeNet can become a stronger and cleaner baseline for hyperspectral image classification research in the future.

## Features
1. Patch-free training and inference
2. Fully End-to-End (w/o preprocess technologies, such as dimension reduction.)

 
## Installation
### 1. Install SimpleCV

```bash
pip install --upgrade git+https://github.com/Z-Zheng/SimpleCV.git
```
## Prepare datasets

### 1. PaviaU
```python
image_mat_path='./pavia/PaviaU.mat'
gt_mat_path='./pavia/PaviaU_gt.mat'
```

### 2. Salinas
```python
image_mat_path='./salinas/Salinas_corrected.mat',
gt_mat_path='./salinas/Salinas_gt.mat',
```

### 3. GRSS2013
```python
# train set
image_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI.tif'
gt_path='./GRSS2013/train_roi.tif'

# val set
image_path='./GRSS2013/2013_IEEE_GRSS_DF_Contest_CASI.tif'
gt_path='./GRSS2013/val_roi.tif'
```


## run experiments

### 1. PaviaU
```bash

bash scripts/freenet_1_0_pavia.sh

```

### 2. Salinas
```bash

bash scripts/freenet_1_0_salinas.sh

```

### 3. GRSS2013
```bash

bash scripts/freenet_1_0_grss.sh

```