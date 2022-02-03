# Installation and use of this study

We experienced some issues with reproducing the exact setup using the original provided versions of libraries. 
In our case, using the latest version of the libraries yielded a working setup.

These instructions will let you install all necessary packages and shows how our high-level notebook can be used to train and evaluate models.

### Clone or download the repository
You can download our latest version from Git with:
`git clone git@github.com:RomanOort/FACTAI.git`

### Conda
We recommend creating a new conda environment.
```
conda create -n fact python=3.9
conda activate fact
```

### Cuda 
Check the version of your cuda toolkit (you will need this later) with:
`nvcc --version`

Example output:
```
(fact) X@X:~$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:01_CDT_2018
Cuda compilation tools, release 10.0, V10.0.130
```

### Pytorch
Select version of pytorch corresponding to your cuda toolkit, see https://pytorch.org/get-started/locally/. \
Install using pip (NOT conda):
`pip3 install torch torchvision torchaudio`

### Pytorch Geometric
Select version of Pytorch Geometric corresponding to your cuda toolkit, see https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
Install using pip (NOT conda):
`pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu102.html`


### Download pretrained models
NOTE: Datasets for Mutagenicty GNN training is not included. The MNIST datasets are downloaded automatically.
Download pretrained models with the following commands:
```
cd FACTAI/packages/gcn_interpretation/gnn-model-explainer/
wget https://modelarts-cnnorth1-market-dataset.obs.cn-north-1.myhuaweicloud.com/example-apps/explain_graphs/ckpt.zip
unzip ckpt.zip
```

### Install auxilary packages
```
cd FACTAI/
pip install -r requirements.txt
cd FACTAI/packages/ldbExtraction/
pip install -e .
```

### Troubleshooting
In case that the above instruction are not sufficient, please see the installation instructions as provided by the original authors.
These can be found in `packages/gcn_interpretation/gnn-model-explainer/INSTALLATION.md`. 

# Using the code
Now that all packages are installed, you can use the code.
We recommend using our high-level notebook to train the models and evaluate them.

The notebook can be found in `FACTAI/packages/gcn_interpretation/gnn-model-explainer/Reproducing Robust Counterfactual Explanation on Graph Neutral Networks.ipynb`.

Please read the cells of the notebook carefully since they are self-explaining and some additional steps have to be taken if you want to download our trained models.

Training a GNN from scratch can also be done from the command line with:
`python train.py --bmname MNISTSuperpixels --gpu --datadir data --num-classes 10 --name-suffix BASE --notes "Larger model, with 4 layers" --hidden-dim 100 --output-dim 30 --pred-hidden-dims [30] --num-gcn-layers 4`



