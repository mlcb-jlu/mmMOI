# mmMOI

# Introduction
A Multi-omics Integration Framework using Multi-label Guided Learning and Multi-scale Fusion

# Download
Download datases and trained model: https://pan.baidu.com/s/13xJdqeqx7jr_uyM7iP8HIg?pwd=sgrc#list/path=%2F&parentPath=%2F

* Our project structure is like:
```
mmMOI
 - MoDF
 - SoRL
 - dataset
 - results
   - data
   - model
```

# Testing

```
cd MoDF

python main.py --dataset GBM --cuda_device 0 --mode False
python main.py --dataset BRCA --cuda_device 0 --mode False
python main.py --dataset OV --cuda_device 0 --mode False
python main.py --dataset KIPAN --cuda_device 0 --mode False

```

# Training
## Single-omics Data Representation
```
cd SoRL

python main.py --dataset GBM --cuda_device 0
python main.py --dataset BRCA --cuda_device 0
python main.py --dataset OV --cuda_device 0
python main.py --dataset KIPAN --cuda_device 0

```
##  Multi-omicsDataFusion
```
cd MoDF

python main.py --dataset GBM --cuda_device 0 --mode True
python main.py --dataset BRCA --cuda_device 0 --mode True
python main.py --dataset OV --cuda_device 0 --mode True
python main.py --dataset KIPAN --cuda_device 0 --mode True

```

