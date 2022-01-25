from scipy.sparse import coo_matrix
import numpy as np
from sklearn.metrics import roc_auc_score

# gt = np.array([
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 1],
# ])
#
# pred = np.array([
#     [1, 1, 0],
#     [0, 0, 0],
#     [0, 0, 0],
# ])

gt = np.zeros((20,20))
pred = np.zeros((20,20))


# Add 10 true positives
gt[0, 10:20] = 1
pred[0, 10:20] = 1

# Add 10 false negatives
gt[1, 10:20] = 1

# Add 10 false positives
pred[2, 10:20] = 1


# 20*20 - 30 = 370 true negatives

def original(pred, gt):
    pred_sparse = coo_matrix(pred)

    reals = []
    preds = []

    for r, c in list(zip(pred_sparse.row, pred_sparse.col)):
        if gt[r, c] != 0 or gt[c, r] != 0:
            reals.append(1)

        else:
            reals.append(0)

        preds.append(pred[r][c])

    # Original work included this this, but its usually not significant due to large number of edges in total dataset
    # reals.append(0)
    # preds.append(0)

    # print("  reals:", reals)
    # print("  preds:", preds)

    return roc_auc_score(reals, preds)

def full_AUC(pred, gt):
    # print("  reals:", gt.ravel())
    # print("  preds:", pred.ravel())
    return roc_auc_score(gt.ravel(), pred.ravel())

def include_fn(pred, gt):
    pred_sparse = coo_matrix(pred)
    gt_sparse = coo_matrix(gt)

    nonzero = set(zip(pred_sparse.row, pred_sparse.col)).union(set(zip(gt_sparse.row, gt_sparse.col)))

    reals = []
    preds = []

    for r, c in nonzero:
        if gt[r, c] != 0 or gt[c, r] != 0:
            reals.append(1)
        else:
            reals.append(0)

        preds.append(pred[r][c])


    # print("  reals:", reals)
    # print("  preds:", preds)

    return roc_auc_score(reals, preds)

print("============= Ground truth:")
print(gt)

print("============= Prediction:")
print(pred)

print()
print("=== END")
print()

score = original(pred, gt)
print("original", score)
print()

score = full_AUC(pred, gt)
print("full", score)
print()

score = include_fn(pred, gt)
print("include FN", score)