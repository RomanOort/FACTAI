import torch
import numpy as np

def process_for_boundary(data, size_l, max_nodes, train_idx, val_idx, width):
    adj_t = torch.zeros((size_l,max_nodes,max_nodes)).float()
    feats_t = torch.zeros((size_l,max_nodes,width)).float()
    label_n = np.zeros((size_l), dtype=np.int32)
    pred_n = np.zeros((size_l), dtype=np.int32)
    new_idx_n = np.zeros((size_l), dtype=np.int32)
    old_idx_n = np.zeros((size_l), dtype=np.int32)
    num_nodes_n = np.zeros((size_l), dtype=np.int32)

    for i in range(len(data['adj'])):
        adj = torch.from_numpy(data['adj'][i]).float()
        feat = torch.from_numpy(data['feat'][i]).float()

        new_idx = data['new_idx'][i]
        old_idx = data['old_idx'][i]
        pred = data['pred'][i][new_idx]
        label = data['label'][i][new_idx]
        nodes = adj.shape[0]

        adj_t[i,:nodes,:nodes] = adj
        feats_t[i,:nodes,:] = feat
        new_idx_n[i] = new_idx
        old_idx_n[i] = old_idx
        label_n[i] = label
        pred_n[i] = pred
        num_nodes_n[i] = nodes

    label_t = torch.from_numpy(label_n).long()
    pred_t = torch.from_numpy(pred_n).long()
    new_idx_t = new_idx_n
    old_idx_t = old_idx_n
    num_nodes_t = num_nodes_n

    train_data = (adj_t[:train_idx], feats_t[:train_idx], label_t[:train_idx], num_nodes_t[:train_idx], new_idx_t[:train_idx], pred_t[:train_idx], old_idx_t[:train_idx])
    val_data = (adj_t[-val_idx:], feats_t[-val_idx:], label_t[-val_idx:], num_nodes_t[-val_idx:], new_idx_t[-val_idx:], pred_t[-val_idx:], old_idx_t[-val_idx:])
    return train_data, val_data