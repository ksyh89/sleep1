import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


def get_preds(X, model):
    """ Computes prediction of observations X."""
    bs = 256
    preds = []
    with torch.no_grad():
        for idx in range(0, len(X), bs):
            x = torch.from_numpy(X[idx: idx + bs].astype(np.float32)).cuda()
            pred = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    return preds


def compute_AUC(y, preds):
    """AUC 계산."""
    y = y.astype(np.long).reshape([-1])
    preds = preds.astype(np.float32).reshape([-1])
    AUC = roc_auc_score(y, preds)
    return AUC


def compute_accuracy(y, preds):
    """정확도 계산."""
    right = (preds > 0.5).astype(int) == y
    accuracy = np.sum(right) / len(preds)
    return accuracy


def plot_AUC(test_dataset, test_preds, test_AUC, savepath="AUC.png"):
    """Validation set에 대한 AUC를 Plot으로 그린다."""
    precision, recall, _ = precision_recall_curve(test_dataset.data[:, :1], test_preds)
    fpr, tpr, _ = roc_curve(test_dataset.data[:, :1], test_preds)
    plt.figure()

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    f.suptitle("AUC %.4f" % test_AUC)
    f.set_size_inches((8, 4))

    axes[0].fill_between(recall, precision, step="post", alpha=0.2, color="b")
    axes[0].set_title("Recall-Precision Curve")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("ROC curve")
    #plt.show()
    plt.savefig(savepath)
    print(savepath)

def plot_AUC_v2(preds_list, target, savepath="AUC.png"):
    plt.figure()

    f, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    f.set_size_inches((6, 6))

    for label, preds in preds_list:
        print(label, preds)
        fpr, tpr, _ = roc_curve(target, preds)
        axes.plot(fpr, tpr, label=label)
    axes.plot([0, 1], [0, 1], linestyle="--")
    axes.legend()
    plt.savefig(savepath)
    print(savepath)
