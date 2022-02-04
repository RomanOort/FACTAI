# Robust Counterfactual Explanations on Graph Neural Networks (RCExplainer)

This code is implemented on top of GNNExplainer's codebase.
 https://github.com/RexYing/gnn-model-explainer

**For information on setup and installation of environment, go to readme of the parent directory.**

## Directory Setup
### Datasets

**Datasets are NOT required separately for inference or for training of the explanation methods. For ease, we have clubbed the datasets with the trained GNN pytorch models. Download trained models from this link.**
 https://drive.google.com/file/d/14Tlv_beU8sGVk22AKsgseRF6dJEkXJAn/view?usp=sharing 
 
 **Place 'ckpt' folder in 'RCExplainer/gcn_interpretation/gnn-model-explainer/'.**

Datasets are ONLY required for training GNN model from scratch and are available for download from the following link if required:
https://drive.google.com/file/d/1k3VjFPvHWM71Lfb9AXjHcC1MlJkMvoX5/view?usp=sharing
For training GNN from scratch, please refer training GNN section. A copy of the datasets should be placed in `RCExplainer/gcn_interpretation/datasets` for this.


### Pre-trained Models 
 Download all trained models from this link:
https://drive.google.com/file/d/14Tlv_beU8sGVk22AKsgseRF6dJEkXJAn/view?usp=sharing
and place 'ckpt' folder in `RCExplainer/gcn_interpretation/gnn-model-explainer`

GNN model + dataset is saved in `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{dataset_name}_base.pth.tar`

Explainer models are saved in `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{method_name}/{method_name}.pth.tar`


## Explaining Trained GNN Models

This section covers how to evaluate already trained GNN models


### Evaluating for fidelity-sparsity

The graph classification datasets are described below. We evaluate fidelity and sparsity on these datasets. 

| Dataset Name | Reference name  | Ckpt dir          | Explainer dir | Relevant class |
| ---          | ---        | ---               | --- | --- |
| BA-2Motifs    | BA_2Motifs       | `ckpt/BA_Motifs_3gc`   | TBD | 1 |
| Mutagenicity | Mutagenicity       | `ckpt/orig_Mutagenicity`   | TBD | 0 |
| NCI1  | NCI1       |`ckpt/NCI3_3gc`    | TBD | 1|

Below are commands for different datasets
1) Mutagenicity
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --exp-path "./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar" --multigraph-class 0 --eval

```

2) BA_2Motifs
```
python explainer_main.py --bmname BA_2Motifs --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/BA_2Motifs" --exp-path "./ckpt/BA_2Motifs/RCExplainer/rcexplainer.pth.tar" --multigraph-class 1 --eval

```

3) NCI1
```
python explainer_main.py --bmname NCI1 --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/NCI1" --exp-path "./ckpt/NCI1/RCExplainer/rcexplainer.pth.tar" --multigraph-class 1 --eval

```

General method for running:

```
python explainer_main.py --bmname {Reference name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --graph-mode --ckptdir {Ckpt dir}  --exp-path  {Explainer dir} --multigraph-class {Relevant class}
```

### Evaluating for noise robustness 

For noise robustness results, simply add the `--noise` flag to the above commands. For example, for graph classification:

```
python explainer_main.py --bmname {Reference name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --graph-mode --ckptdir {Ckpt dir}  --exp-path  {Explainer dir} --multigraph-class {Relevant class} --noise
```

Example:
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --exp-path "./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar" --multigraph-class 0 --eval

```

### Evaluating node classification for accuracy and AUC

The node classification datasets are described below. We refer to `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt` as `ckpt`

| Dataset Name | Reference name  | Ckpt dir          | Explainer dir |AUC | Node Accuracy |
| ---          | ---        | ---               | --- |--- | --- |
| BA-Shapes    | syn1       | `ckpt/syn1`   | TBD |0.998 | 0.973 |
| BA-Community | syn2       | `ckpt/syn2`   | TBD |0.995 | 0.916 |
| Tree-Grid  | syn3       |`ckpt/syn3`    | TBD |0.993 | 0.993 |
| Tree-Cycles   | syn4       |`ckpt/syn4`    | TBD |0.995 | 0.974 |


Below are commands for node classification datasets 

1) syn1 dataset (BA-Shapes)
```
python explainer_main.py --bmname syn1 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn1" --exp-path "./ckpt/syn1/RCExplainer/rcexplainer.pth.tar" --eval
```


2) syn2 dataset (BA-Community)
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn2" --exp-path "./ckpt/syn2/RCExplainer/rcexplainer.pth.tar" --eval
```

3) syn3 dataset (Tree-Grid)
```
python explainer_main.py --bmname syn3 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn3" --exp-path "./ckpt/syn3/RCExplainer/rcexplainer.pth.tar" --eval
```


4) syn4 dataset (Tree-Cycles)
```
python explainer_main.py --bmname syn4 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn4" --exp-path "./ckpt/syn4/RCExplainer/rcexplainer.pth.tar" --eval
```

General method for running:

```
python explainer_main.py --bmname {Dataset name} --num-gc-layers 3 --explainer-method rcexplainer --gpu --eval --ckptdir {Ckpt dir}  --exp-path  {Explainer dir}
```


## Training Explainers

To train an explainer, you will have to pass in appropriate hyperparameters for each method.

For RCExplainer, this may include the learning rate `--lr 0.001`, the boundary and inverse boundary coefficient (which make up lambda) `--boundary_c 3 --inverse_boundary_c 12`, the size coefficient `--size_c 0.01`, the entropy coefficient `--ent_c 10`, and the boundary loss version `--bloss-version "sigmoid"` and name of folder  where logs and trained models should be stored `--prefix training_dir`. 
The logs and models will be saved at `RCExplainer/gcn_interpretation/gnn-model-explainer/ckpt/{dataset_name}/{prefix}/`


### Graph classification
Below are commands for training 

1) Mutagenicity
```
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```

2) BA_2Motifs
```
python explainer_main.py --bmname BA_2Motifs --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/BA_2Motifs" --multigraph-class 1 --prefix "rcexp_ba2motifs" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```

3) NCI1
```
python explainer_main.py --bmname NCI1 --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/NCI1" --multigraph-class 1 --prefix "rcexp_nci1" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --epochs 601

```


###  Node classification

Below are commands for training


Training explainer on syn2 (BA-Community) 
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --lr 0.001 --boundary_c 10 --inverse_boundary_c 5 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --topk 4 --prefix test-rc-syn2 --ckptdir "./ckpt/syn2_emb/"
```

1) syn1 dataset (BA-Shapes)
```
python explainer_main.py --bmname syn1 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn1 --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn1" --epochs 601
```


2) syn2 dataset (BA-Community)
```
python explainer_main.py --bmname syn2 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn2"  --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn2" --epochs 601
```

3) syn3 dataset (Tree-Grid)
```
python explainer_main.py --bmname syn3 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn3"  --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn3" --epochs 601
```


4) syn4 dataset (Tree-Cycles)
```
python explainer_main.py --bmname syn4 --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/syn4" --lr 0.001 --boundary_c 12.0 --inverse_boundary_c 3.0 --size_c 0.09 --ent_c 10.0 --bloss-version "sigmoid" --prefix test-rc-syn4" --epochs 601
```

## Other baselines
For training other baselines, replace ``--explainer-method rcexplainer`` in above commands with ``--explainer-method {baseline}`` where baseline can be picked from these methods  gnnexplainer | pgexplainer | pgmexplainer | rcexp_noldb. 

Some of the trained baselines and info can be downloaded from this url:
https://drive.google.com/file/d/1t1i8kNjtGkqhI43ehGFvlxvbEan7HL80/view?usp=sharing
For inference on these methods, simply change --exp-path in above inference commands to the provide explainer models. Inference of pgexplainer and rcexp_noldb baseline models is supported by rcexplainer, so keep the --explainer-method rcexplainer in inference commands. 


## Training GNN models from scratch

To train the GNN models from scratch, more setup is required:

### Dependencies

It is necessary to install [pytorch geometric][pytorch-geometric.readthedocs.io/en/latest/notes/installation.html] matching your CUDA (10.1) and PyTorch version (1.8.0). A model can be trained on a node classification dataset by:

```
python train.py --dataset syn1 --gpu
```

A model can be trained on a graph classification dataset by:

```
python train.py --bmname BA_2Motifs --gpu
```

For more details on training GNN models from scratch, check this GNNExplainer url
https://github.com/RexYing/gnn-model-explainer

