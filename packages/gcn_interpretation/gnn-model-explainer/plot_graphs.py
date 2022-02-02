import json
import numpy as np 
import matplotlib.pyplot as plt
import argparse
import os

def load_data(path):
    """
    Get data from the specified file.
    """
    with open(path) as file:
        return json.load(file)
    
def compute_avg_std(x,y):
    """
    Compute average of x and y.
    Compute std of y.
    """
    std = np.array(y).std(axis=0)
    avg_x = np.array(x).mean(axis=0)*100
    avg_y = np.array(y).mean(axis=0)
    
    return avg_x, avg_y, std
    
def compute_values(data, x_name, y_name):
    """
    Compute avg and std of train and test results.
    """
    x_train, x_test, y_train, y_test = [], [], [], []
    
    for value in data.values():
        x_train.append(value['train'][x_name])
        y_train.append(value['train'][y_name])
        
        x_test.append(value['test'][x_name])
        y_test.append(value['test'][y_name])
        
    train = compute_avg_std(x_train, y_train)
    test = compute_avg_std(x_test, y_test)
        
    return train, test

def plot(data, labels, model_name, axes, sp , threshold_sp, type_data, type='same'):
    """
    Plots the values given in data, with the corresponding labels and titles.
    """
    colors = {'RCExplainer': 'g', 'PGExplainer': 'b', 'RCExp-NoLDB': 'r', 'Pretrained RCExplainer': 'k'}
    plotting_dict = {'train':   {'label': model_name + ': train',   'color': colors[model_name]},
                     'test':    {'label': model_name + ': test',    'color': '--'+ colors[model_name]},
                     'same':    {'label': model_name,               'color': colors[model_name]}}
    
    x, y, std = data
    # We only look at sparsity values higher than some threshold 
    mask_x = np.where(y >= threshold_sp)[0] if type_data == 'fidelity' else np.arange(len(x)) 
    
    axes.errorbar(x[mask_x], y[mask_x], yerr=std[mask_x], capsize=4, label=plotting_dict[type]['label'], fmt=plotting_dict[type]['color'], linewidth=0.75)
    axes.set_title(f"Train/test split: 80/20%") if sp == 0.8 else axes.set_title(f"Train/test split: 100/100%")
    axes.set_xlabel(labels[0] + " (%)")
    axes.set_ylabel(labels[1])
    axes.legend(fontsize=6)
    
    # Set axis to match paper
    if type_data == 'fidelity':
        axes.set_yticks(np.arange(0, .71, 0.1))
    elif type_data == 'noise':
        axes.set_xticks(np.arange(0, 31, 10))
        axes.set_yticks(np.arange(0.5, 1.01, 0.1))

def plot_main(results_folder, type_data, figsize, threshold=0):
    models = os.listdir(results_folder)
    sparsity = [0.8, 1.0]
    names = {'rcexp': 'RCExplainer', 'pgexplainer': 'PGExplainer', 'rcexp_noldb': 'RCExp-NoLDB', 'pretrained_rcexplainer': 'Pretrained RCExplainer'}
    
    _, (ax1, ax2) = plt.subplots(1,2, figsize=figsize)
    
    for model in models:
        for i, sp in enumerate(sparsity):
            data = load_data(f"{results_folder}/{model}/{type_data}_{sp}.json")
            
            if data == {}: continue
            if model == "pretrained_rcexplainer" and sp == 0.8: continue

            if type_data == 'fidelity':
                labels = ['sparsity', 'fidelity', 'Sparsity', 'Fidelity']
            elif type_data == 'noise':
                labels = ['noise_level', 'roc_auc', 'Noise', 'AUC']
                
            train, test = compute_values(data, labels[0], labels[1])
            
            if sp == 0.8:
                plot(train, labels[2:], names[model], ax1, sp, threshold, type_data, type='train')
                plot(test, labels[2:], names[model], ax1, sp, threshold, type_data, type='test')
            elif sp == 1.0:
                plot(test, labels[2:], names[model], ax2, sp, threshold, type_data, type='same')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # python3 plot_graphs.py
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='path to ‘results’ folder', default='results')
    args = parser.parse_args()
    
    plot_main(args.path, 'fidelity', (10,2), threshold=0)
    plot_main(args.path, 'noise', (10,2))