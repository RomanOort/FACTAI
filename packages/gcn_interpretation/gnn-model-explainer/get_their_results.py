import os
from get_results import load_data, save_data, store_fidelity, store_roc
from types import SimpleNamespace
from explainer_main import main

SEEDS = [0, 1, 3, 5, 8, 10, 15, 42, 69, 101]
SPARSITIES = [0.8, 1.0]


config = {
    "logdir":"log",
    "ckptdir": "ckpt/Mutagenicity",
    "prefix":"",
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
    "train_data_sparsity": 0.8,
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
    "apply_filter": False,
    "writer": True,
    "bias": True,
    "clean_log": False,
    "exp_path": "./ckpt/Mutagenicity/RCExplainer/rcexplainer.pth.tar"
}


def get_train_test_results(seed, sparsity):
    config["seed"] = int(seed)
    config["train_data_sparsity"] = float(sparsity)

    args = SimpleNamespace(**config)  # Namespace from dict
    train, test = main(args)

    return train, test


if __name__ == "__main__":
    results_dir = "results/"
    model = "pretrained_rcexplainer"

    # Create results folder
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/{model}", exist_ok=True)

    for data_sparsity in SPARSITIES:
        print("Evaluating sparsity", data_sparsity)

        fid_path = f"{results_dir}/{model}/fidelity_{data_sparsity}.json"
        noise_path = f"{results_dir}/{model}/noise_{data_sparsity}.json"
        fid = load_data(fid_path)
        noise = load_data(noise_path)
        for seed in SEEDS:
            print("Evaluating seed", seed)

            if seed in fid and seed in noise:
                print("Already evaluated", model, seed, data_sparsity)
                continue

            train, test = get_train_test_results(seed, data_sparsity)
            if train is None and test is None:
                continue

            store_fidelity(train, test, str(seed), fid)
            store_roc(train, test, str(seed), noise)

        save_data(fid, fid_path)
        save_data(noise, noise_path)
