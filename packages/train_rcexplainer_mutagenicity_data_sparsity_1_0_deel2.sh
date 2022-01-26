python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=5 &
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=8
wait
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=69 &
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=101