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
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--path', type=str, nargs='+',
    #                     help='path to ‘results’ folder')
    #
    # args = parser.parse_args()
    #
    # plot_graphs(args.path[0], 'fidelity')
    # plot_graphs(args.path[0], 'noise')


    rc_SPARSITY = [
        0.0, 0.5005545413557063, 0.7506549526336842, 0.8008183883865235, 0.8507366861023127, 0.9007882990061064, 0.9507338678642242, 0.990804896810223, 1.0]
    rc_FIDELITY = [
        0.9968429568348398, 0.9964667587499463, 0.9943339327279616, 0.9892777030821666, 0.9792143489183627, 0.9372607581728762, 0.6766841419388111, 0.05507931876900788, 0.0]

    rc_AUC = [
        0.9848490720797608, 0.8419735591755685, 0.6996266633321102, 0.5949986179887606, 0.4928386834077074, 0.4650387377005101, 0.47602118051353715]

    # RCExplainer with 200 epochs to check overfitting
    rc_SPARSITY = [
        0.0, 0.5005545413557063, 0.7506549526336842, 0.8008183883865235, 0.8507366861023127, 0.9007882990061064, 0.9507338678642242, 0.990804896810223, 1.0]
    rc_FIDELITY = [
        0.9968429568348398, 0.9964667587499463, 0.9943339327279616, 0.9892777030821666, 0.9792143489183627, 0.9372607581728762, 0.6766841419388111, 0.05507931876900788, 0.0]
    rc_AUC = [
        0.9848490720797608, 0.7521247225163356, 0.6014714582593378, 0.54119596934485, 0.5073962508896175, 0.4836306752326382, 0.4799958730963047]

    pg_SPARSITY = [
        0.0, 0.5005564204901385, 0.7506547995587696, 0.8008181305098752, 0.8507368665615984, 0.9007886357451653, 0.9507330647610267, 0.9908026036481358, 1.0]
    pg_FIDELITY = [
        0.9968429568348398, 0.9965818690469617, 0.9954166726257958, 0.9910404799408599, 0.9806262115289315, 0.9427061302697378, 0.6915596229164176, 0.058334323294564505, 0.0]
    pg_AUC = [
        0.9962792508797925, 0.9406341508899838, 0.8985420845622384, 0.8795713699560317, 0.8668008226866173, 0.8562583273563397, 0.8500991741076628]

    S = 3
    fig, axs = plt.subplots(1,2, figsize=(S*3,S),dpi=200)
    axs[0].set_title("Fidelity performance")

    axs[0].plot(np.array(rc_SPARSITY) * 100, rc_FIDELITY,  label="RCExplainer", color="green")
    axs[0].plot(np.array(pg_SPARSITY) * 100, pg_FIDELITY,  alpha=.7,  label="PGExplainer", color="blue", )

    axs[0].set_xlim(70, 100)
    axs[0].set_xlabel("Sparsity (%)")
    axs[0].set_ylabel("Fidelity")


    axs[1].set_title("Noise robustness")

    NOISE_VALS = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]) * 100
    axs[1].plot(NOISE_VALS, rc_AUC, label="RCExplainer", color="green")
    axs[1].plot(NOISE_VALS, pg_AUC, alpha=.7, label="PGExplainer", color="blue")

    axs[1].set_xlabel("Noise (%)")
    axs[1].set_ylabel("AUC")
    axs[0].legend(loc="lower left")
    axs[1].legend()
    plt.tight_layout()

    plt.show()
    
    
    
    