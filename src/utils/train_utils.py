import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score

import PIL

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


def compute_PRAUC(y,preds):
    """PRAUC 계산."""
    y = y.astype(np.long).reshape([-1])
    preds = preds.astype(np.float32).reshape([-1])
    PRAUC = average_precision_score(y, preds)
    return PRAUC


def compute_accuracy(y, preds):
    """정확도 계산."""
    right = (preds > 0.5).astype(int) == y
    accuracy = np.sum(right) / len(preds)
    return accuracy

def compute_confusion(y, preds):
    """confusion matrix 계산."""
    Truepositive = ((preds > 0.5).astype(int) == 1) & (y == 1)
    TP = np.sum(Truepositive)

    Truenegative = ((preds > 0.5).astype(int) == 0) & (y == 0)
    TN = np.sum(Truenegative)

    Falsenegative = ((preds > 0.5).astype(int) == 0) & (y == 1)
    FN = np.sum(Falsenegative)

    Falsepositive = ((preds > 0.5).astype(int) == 1) & (y == 0)
    FP = np.sum(Falsepositive)

    return TP, TN, FN, FP


def plot_AUC(test_dataset, test_preds, test_AUC, savepath="AUC.tiff"):
    """Validation set에 대한 AUC를 Plot으로 그린다."""
    precision, recall, _ = precision_recall_curve(test_dataset.data[:, :1], test_preds)
    fpr, tpr, _ = roc_curve(test_dataset.data[:, :1], test_preds)
    plt.figure()

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    #f.suptitle("AUC %.4f" % test_AUC)
    f.set_size_inches((16, 8))

    axes[0].fill_between(recall, precision, step="post", alpha=0.2, color="dimgrey")
    axes[0].set_title("Recall-Precision Curve", fontsize =14)
    axes[0].set_ylabel('Precision', fontsize=12)
    axes[0].set_xlabel('Recall', fontsize=12)

    axes[1].plot(fpr, tpr, color = 'r', label = "Deep learning (AUC 0.870)" )
    axes[1].plot([0, 1], [0, 1], linestyle="--", color = 'k')
    axes[1].set_title("Receiver Operating Characteristic Curve", fontsize =14)
    axes[1].set_ylabel('True positive rate', fontsize=12)
    axes[1].set_xlabel('False positive rate', fontsize=12)
    axes[1].legend(loc="lower right", fontsize = 12)

    plt.savefig(savepath, dpi=300)
    plt.show()
    print(savepath)

def plot_AUC_v2(preds_list, target, savepath="AUC.tiff"):
    plt.figure()

    f, axes = plt.subplots(1, 1, sharex=True, sharey=True)
    f.set_size_inches((8, 8))

    for label, preds in preds_list:
        print(label, preds)
        fpr, tpr, _ = roc_curve(target, preds)
        if (label == 'Deep learning (AUC 0.870)'):
            red = 'r'
        if (label == 'Logistic regression (AUC 0.858)'):
            red = 'orange'
        if (label == 'Linear SVM (AUC 0.849)'):
            red = 'cornflowerblue'
        if (label == 'Random forest classifier (AUC 0.810)'):
            red = 'green'
        if (label == 'K-nearest neighbors (AUC 0.740)'):
            red = 'purple'

        axes.plot(fpr, tpr, label=label, color=red)

    axes.plot([0, 1], [0, 1], linestyle="--", color = 'k')
    axes.set_title("Receiver Operating Characteristic Curve", fontsize =14)
    axes.set_xlabel('False positive rate', fontsize=12)
    axes.set_ylabel('True positive rate', fontsize=12)
    axes.legend(loc="lower right", fontsize = 12)
    plt.savefig(savepath, dpi=300)
    plt.show()  
    print(savepath)
