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

noise_range = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

Original= np.array([0.9908604110735414, 0.9511666705401967, 0.8969179466951889, 0.8475323165197972, 0.8051372667666594, 0.7621745534807047, 0.7367770322271583])


FN= np.array([0.9908604110735414, 0.950301344238759, 0.8941436371803021, 0.8429060484165232, 0.799834142848, 0.7559895125263308, 0.7301814299735965])


Full = np.array([0.9999730228798774, 0.9993951978190219, 0.9981218106302389, 0.9967593888193426, 0.996027002366392, 0.9950794348599054, 0.9945387911727037])

diff = (1- (FN / Original)) * 100





def fts(x):
    return f"{x:.4f}"


print('l'*(len(Original)+1))

print(" \\toprule")
print("Noise level & ", " & ".join(map(str, noise_range)), "\\\\")
print(" \\midrule")

print("All (FN+FP+TN+TP) &", " & ".join(map(fts, Full)), "\\\\")
print("Original &", " & ".join(map(fts, Original)), "\\\\")
print("False negatives &", " & ".join(map(fts, FN)), "\\\\")

def fts_pct(x):
    return f"{x:.3f}\%"
print("full and FN diff &", "& ".join(map(fts_pct, diff)), "\\\\")
print(" \\bottomrule")



print("---------------- PG Explainer")

# PG Explainer
Original= np.array([0.9279181609268776, 0.8809680598039797, 0.8293182050575028, 0.784613966051129, 0.7486899651888039, 0.7178781594272209, 0.6941494930746562])
FN =      np.array([0.9279181609268776, 0.8799844776438122, 0.8265048443189288, 0.780857252912548, 0.7425332661728661, 0.7105413906910457, 0.6863249416229291])
Full =    np.array([0.9996371917867928, 0.9988296918249906, 0.9974007349935546, 0.9964389589668325, 0.9944888531005899, 0.9932832678351772, 0.9925871054285493])
diff = (1- (FN / Original)) * 100

print(" \\toprule")
print("Noise level & ", " & ".join(map(str, noise_range)), "\\\\")
print(" \\midrule")

print("All (FN+FP+TN+TP) &", " & ".join(map(fts, Full)), "\\\\")
print("Original &", " & ".join(map(fts, Original)), "\\\\")
print("False negatives &", " & ".join(map(fts, FN)), "\\\\")

def fts_pct(x):
    return f"{x:.3f}\%"
print("Original and false negatives difference &", "& ".join(map(fts_pct, diff)), "\\\\")
print(" \\bottomrule")


if __name__ == '__main__' and False:
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