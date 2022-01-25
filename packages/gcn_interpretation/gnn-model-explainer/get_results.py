import os
from types import SimpleNamespace
from explainer_main import main
import json


SEEDS = [0, 1, 3, 5, 8, 10, 15, 42, 69, 101]
SPARSITIES = [0.8, 1.0]
MODELS = ["rcexp", "rcexp_noldb", "pgexplainer"]


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


def store_roc(train, test, seed: str, file):
    if seed not in file:
        file[seed] = {'train': dict(), 'test': dict()}

    s_train, f_train, n_train, r_train = train
    s_test, f_test, n_test, r_test = test

    file[seed]['train']['noise_level'] = n_train
    file[seed]['test']['noise_level'] = n_test
    file[seed]['train']['roc_auc'] = r_train
    file[seed]['test']['roc_auc'] = r_test


def store_fidelity(train, test, seed: str, file):
    if seed not in file:
        file[seed] = {'train': dict(), 'test': dict()}

    s_train, f_train, n_train, r_train = train
    s_test, f_test, n_test, r_test = test

    file[seed]['train']['fidelity'] = f_train
    file[seed]['test']['fidelity'] = f_test
    file[seed]['train']['sparsity'] = s_train
    file[seed]['test']['sparsity'] = s_test


def get_file_name(model: str, seed: str, sparsity: str, dataset='mutag'):
    path = f"saved_models/{model}_{dataset}_seed_{seed}_sparsity" \
           f"_{sparsity}_logdir/{model}_mutagexplainer_Mutagenicity_ep_600_seed_" \
           f"{seed}_sparsity_{sparsity}.pth.tar"
    return path


def get_train_test_results(model, seed, sparsity):
    f_path = get_file_name(model, seed, sparsity)
    if not os.path.isfile(f_path):
        print("Skipping", model, "with seed", seed, "and sparsity", sparsity)
        print("Cannot find model in", f_path)
        return None, None

    config["seed"] = int(seed)
    config["exp_path"] = f_path
    config["train_data_sparsity"] = float(sparsity)

    args = SimpleNamespace(**config)  # Namespace from dict
    train, test = main(args)

    return train, test


if __name__ == "__main__":
    results_dir = "results/"

    # Create results folder
    os.makedirs(results_dir, exist_ok=True)

    for model in MODELS:
        print("Evaluating model", model)
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

                train, test = get_train_test_results(model, seed, data_sparsity)
                if train is None and test is None:
                    continue

                store_fidelity(train, test, str(seed), fid)
                store_roc(train, test, str(seed), noise)

            save_data(fid, fid_path)
            save_data(noise, noise_path)
