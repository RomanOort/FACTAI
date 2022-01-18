import os
from types import SimpleNamespace
from explainer_main import main
import json


SEEDS = [0, 1, 3, 5, 8, 10, 15, 42, 69, 101]

config = {
    "logdir":"log",
    "ckptdir": "ckpt/Mutagenicity",
    "prefix":"",
    "exp_path":"ckpt/Mutagenicity/rcexp_mutag_seed_1_logdir/rcexp_mutag_seed_1explainer_Mutagenicity_pgeboundary.pth.tar",
    "add_self":"none",
    "dataset":"Mutagenicity",
    "opt":"adam",
    "opt_scheduler":"none",
    "cuda":"1",
    "lr":0.1,
    "clip": 2.0,
    "batch_size": 20,
    "num_epochs": 100,#100,
    "hidden_dim": 20,
    "output_dim": 20,
    "start_epoch": 0,
    "dropout": 0.0,
    "method": "base",
    "name_suffix": "",
    "explainer_suffix": "",
    "align_steps":1000,
    "explain_node":None,
    "graph_idx":-1,
    "mask_act":"sigmoid",
    "multigraph_class":0,
    "multinode_class":-1,
    "add_embedding": False,
    "size_c": -1.0,
    "size_c_2": -1.0,
    "lap_c": -1.0,
    "boundary_c": 0.5,
    "pred_hidden_dim": 20,
    "pred_num_layers": 0,
    "bn": False,
    "train_data_sparsity": 1.0,
    "draw_graphs": False,
    "inverse_noise": False,
    "gumbel": False,
    "node_mask": False,
    "post_processing": False,
    "inverse_boundary_c": 0.5,
    "sparsity": 0.5,
    "ent_c": -1.0,
    "ent_c_2": -1.0,
    "intersec_c": -1.0,
    "topk": 8.0,
    "noise_percent": 10.0,
    "fname": "",
    "bloss_version": "",
    "bmname": "Mutagenicity",
    "graph_mode": True,
    "num_gc_layers": 3,
    "explainer_method": "rcexplainer",
    "gpu": True,
    "eval": True,
    "noise": True,
    "apply_filter": True,
    "writer": True,
    "bias": True,
    "clean_log": False,
}


def save_data(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def load_data(path):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)

    return data


def store_fid(fid, spar, seed, file):
    seed = str(seed)
    if seed not in file:
        file[seed] = dict()

    file[seed]['fidelity'] = fid
    file[seed]['sparsity'] = spar


def store_auc(noise, auc, seed, file):
    seed = str(seed)
    if seed not in file:
        file[seed] = dict()

    file[seed]['noise_level'] = noise
    file[seed]['roc_auc'] = auc


if __name__ == "__main__":
    results_dir = "results/"
    fidelity_path = "fidelity.json"
    roc_path = "roc_auc.json"

    # Create results folder
    os.makedirs(results_dir, exist_ok=True)

    fidelity_res = load_data(results_dir + fidelity_path)
    roc_auc_res = load_data(results_dir + roc_path)

    for seed in SEEDS[:2]:
        print("Evaluating seed", seed)
        if str(seed) in fidelity_res and str(seed) in roc_auc_res:
            continue

        config["seed"] = seed
        config["exp_path"] = f"ckpt/Mutagenicity/rcexp_mutag_seed_{seed}_logdir/rcexp_mutag_seed_{seed}explainer_Mutagenicity_pgeboundary.pth.tar"
        if not os.path.exists(config["exp_path"]):
            raise IOError("Seed file not available", config["exp_path"])

        args = SimpleNamespace(**config)  # Namespace from dict
        sparsity, fidelity, noise_level, roc_auc = main(args)

        store_fid(fidelity, sparsity, seed, fidelity_res)
        store_auc(noise_level, roc_auc, seed, roc_auc_res)

    save_data(fidelity_res, results_dir + fidelity_path)
    save_data(roc_auc_res, results_dir + roc_path)
