# FreeNet
By Zhuo Zheng, Yanfei Zhong, Ailong Ma and Liangpei Zhang

This is an official implementation of FreeNet in our paper ["FPGA: Fast Patch-Free Global Learning Framework for Fully End-to-End Hyperspectral Image Classification"]().


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