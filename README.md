# 3DAD
## Requirements

### Installation
```bash
conda create -n 3dad python=3.8
conda activate 3dad

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit
# e.g., conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3

pip install -r requirements.txt
```
Install GPU-related packages:
```bash
# Chamfer Distance and EMD
cd ./extensions/chamfer_dist
python setup.py install --user
cd ../emd
python setup.py install --user

# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# GPU kNN
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```
## Dataset
The MvTec 3D-AD can be downloaded from [here](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad/downloads/).

## Training and Evaluating
teacher pretraining:
```bash
python test.py
```

student training:
```bash
python kd.py
```

## Acknowledgment

We thank the authors of following works for opening source their excellent codes.  
* [PointNet/PointNet++](https://github.com/charlesq34/pointnet2), [DGCNN](https://github.com/WangYueFt/dgcnn)
* [Point-BERT](https://github.com/lulutang0608/Point-BERT)
* [Point-M@AE](https://github.com/ZrrSkywalker/Point-M2AE)
We also thank the authors of related papers/repos for their inspiring discussions with us.
