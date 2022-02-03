# Environment setup instructions
We experienced a lot of issues with reproducing the exact setup using the original provided versions of libraries. In our case, using the latest version of the libraries yielded a working setup. 


## Clone or download the repository
git clone git@github.com:RomanOort/FACTAI.git

## We recommend creating a new conda environment
conda create -n fact python=3.9
conda activate fact

## Check cuda toolkit
nvcc --version

Example output:
(fact) twiggers@robolabws6:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130



## Select version of pytorch corresponding to your cuda toolkit, and install using pip (NOT conda)
https://pytorch.org/get-started/locally/
pip3 install torch torchvision torchaudio

## Select version of pytorch corresponding to your cuda toolkit, and install using pip (NOT conda)
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html


## Download pretrained models
NOTE: Datasets for Mutagenicty GNN training is not included. The MNIST datasets are downloaded automatically.

cd FACTAI/packages/gcn_interpretation/gnn-model-explainer/

wget https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/example-apps/explain_graphs/ckpt.zip

unzip ckpt.zip

## Install auxilary packages

cd FACTAI/
pip install -r requirements.txt



# Train explainer
python train.py --bmname MNISTSuperpixels --gpu --datadir data --num-classes 10 --name-suffix BASE --notes "Larger model, with 4 layers" --hidden-dim 100 --output-dim 30 --pred-hidden-dims [30] --num-gcn-layers 4



