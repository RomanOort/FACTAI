import json
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import os

def load_data(path):
    with open(path) as file:
        return json.load(file)
    
def compute_values(data, x_name, y_name, train=True):
    avg_x = []
    avg_y = []
    
    for value in data.values():
        if train:
            avg_x.append(value['train'][x_name])
            avg_y.append(value['train'][y_name])
        else:
            avg_x.append(value['test'][x_name])
            avg_y.append(value['test'][y_name])
        
    std = np.array(avg_y).std(axis=0)
    avg_x = np.array(avg_x).mean(axis=0)*100
    avg_y = np.array(avg_y).mean(axis=0)
    
    return avg_x, avg_y, std

def plot(data, labels, model_name, index, sparsity, mask=False, train=True):
    if train:
        colors = {'RCExplainer': 'g', 'PGExplainer': 'b', 'RCExp-NoLDB': 'r', 'Pretrained RCExplainer': 'k'}
    else:
        colors = {'RCExplainer': '--g', 'PGExplainer': '--b', 'RCExp-NoLDB': '--r', 'Pretrained RCExplainer': '--k'}

    x, y, std = data
    plt.subplot(1,2,index)
    
    # We only look at sparsity values higher than 40 (or other number)
    mask_x = np.where(y >= 0)[0] if mask else np.arange(len(x))
    
    if sparsity == 0.8 and train:
        plt.errorbar(x[mask_x], y[mask_x], yerr=std[mask_x], capsize=4, label=model_name + ': train', fmt=colors[model_name])
    elif sparsity == 0.8:
        plt.errorbar(x[mask_x], y[mask_x], yerr=std[mask_x], capsize=4, label=model_name + ': test', fmt=colors[model_name])
    else:
        plt.errorbar(x[mask_x], y[mask_x], yerr=std[mask_x], capsize=4, label=model_name, fmt=colors[model_name])
        
    plt.rcParams["figure.figsize"] = (20,5)
        
    plt.xlabel(labels[0] + " (%)")
    plt.ylabel(labels[1])
    plt.title(f"Data sparsity: {sparsity}")
    plt.legend(fontsize=6)

def plot_graphs(results_folder, type_data):
    models = os.listdir(results_folder)
    sparsity = [0.8, 1.0]
    names = {'rcexp': 'RCExplainer', 'pgexplainer': 'PGExplainer', 'rcexp_noldb': 'RCExp-NoLDB', 'pretrained_rcexplainer': 'Pretrained RCExplainer'}
    
    for model in models:
        for i, sp in enumerate(sparsity):
            data = load_data(f"{results_folder}/{model}/{type_data}_{sp}.json")
            
            if data == {}:
                continue
            
            if type_data == 'fidelity':
                labels = ['sparsity', 'fidelity', 'Sparsity', 'Fidelity']
                mask = True
            elif type_data == 'noise':
                labels = ['noise_level', 'roc_auc', 'Noise', 'AUC']
                mask = False
                
            train = compute_values(data, labels[0], labels[1], train=True)
            test = compute_values(data, labels[0], labels[1], train=False)
            
            plot(train, labels[2:], names[model], i+1, sp, mask=mask, train=True)
            plot(test, labels[2:], names[model], i+1, sp, mask=mask, train=False) if sp == 0.8 else 0

    plt.show()

if __name__ == '__main__':
    # python3 plot_graphs.py --path=results
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, nargs='+',
                        help='path to ‘results’ folder')

    args = parser.parse_args()
    
    plot_graphs(args.path[0], 'fidelity')
    plot_graphs(args.path[0], 'noise')
    
    
    
    
    
    
    
    
    