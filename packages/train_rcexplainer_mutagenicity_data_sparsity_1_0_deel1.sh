python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=42 &
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=15
wait
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=10 &
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=3
wait
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=0 &
python explainer_main.py --bmname Mutagenicity --graph-mode --num-gc-layers 3 --explainer-method rcexplainer --gpu --ckptdir "./ckpt/Mutagenicity" --multigraph-class 0  --prefix "rcexp_mutag" --lr 0.001 --size_c 0.001 --ent_c 8.0 --boundary_c 3 --inverse_boundary_c 12 --bloss-version "sigmoid" --train-data-sparsity 1.0 --epochs 601 --seed=1