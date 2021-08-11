# PointNet


## Install

### 1. Clone code
```
git clone https://github.com/yahuiliu99/PointNet.git
cd PointNet
```

### 2. Install dependence python packages
Use Anaconda
```
conda create -n <env_name> python=3.7
conda activate <env_name>
pip install -r requirements.txt
```

## Dataset
This code implements object classification on ModelNet10 dataset. DownLoad the Dataset ModelNet10
```
wget http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip
unzip -q ModelNet10.zip
```

## Usage

### Train
#### train with single GPU or CPU

change `args.py` argument `is_dist=False`
```
python train.py
```

#### train with multiple GPUs (Using DistributedDataParallel)

change `args.py` argument `is_dist=True`
```
python -m torch.distributed.launch --nproc_per_node=<num of GPUs> train.py
```
Note: The batch_size in `args.py` is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. 

if you use 4 GPUs, you need to divide batch_size by 4.

### Evaluate
```
python test.py
```

#### Pretrained model
You can use pretrained models provided in this directory. 

## Acknowledgement
This code baseline is borrowed from [pointnet](https://github.com/nikitakaraevv/pointnet) and some errors have been corrected.
