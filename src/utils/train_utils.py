import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve

"""
def get_preds(X, model):
    # Computes prediction of observations X.
    bs = 256
    preds = []
    with torch.no_grad():
        for idx in range(0, len(X), bs):
            x = torch.from_numpy(X[idx: idx + bs].astype(np.float32)).cuda()
            pred = torch.sigmoid(model(x)).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    return preds
"""
def get_preds(X, model):
    """ Computes prediction of observations X."""
    bs = 256
    preds = []
    with torch.no_grad():
        for idx in range(0, len(X), bs):
            x = torch.from_numpy(X[idx: idx + bs].astype(np.float32)).cuda()
            pred = model(x).cpu().numpy()
            preds.append(pred)
    preds = np.concatenate(preds, axis=0)
    return preds

"""
def compute_AUC(y, preds):
    #AUC 계산.
    y = y.astype(np.long).reshape([-1])
    preds = preds.astype(np.float32).reshape([-1])
    AUC = roc_auc_score(y, preds)
    return AUC
"""
def compute_AUC(y, preds):
    """AUC 계산."""
    y = y.astype(np.long)#.reshape([-1])
    preds = preds.astype(np.float32)#.reshape([-1])
    AUC = roc_auc_score(y, preds)
    return AUC

def compute_AUC_micro(y, preds):
    """AUC 계산."""
    y = y.astype(np.long)#.reshape([-1])
    preds = preds.astype(np.float32)#.reshape([-1])
    #AUC_micro = roc_auc_score(y, preds, average='micro')

    test_label_onehot = y
    test_preds = preds
    n_class = int(y.shape[1])
    print(f'n_class is {n_class}')
    
    fpr = dict()
    tpr = dict()
    roc_auc = []
    AUC_micro = []
    from sklearn.metrics import auc
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(test_label_onehot[:, i], test_preds[:, i])
        roc_auc.append(auc(fpr[i], tpr[i]))

    for i in range(n_class):
        AUC_micro.append(roc_auc_score(test_label_onehot[:, i], test_preds[:, i]))

    print(f'np.array(roc_auc).shape is {np.array(roc_auc).shape}')
    print(f'np.array(AUC_micro).shape is {np.array(AUC_micro).shape}')

    return roc_auc

"""
def compute_accuracy(y, preds):
    #정확도 계산.
    right = (preds > 0.5).astype(int) == y
    accuracy = np.sum(right) / len(preds)
    return accuracy
"""
def compute_accuracy(y, preds):
    """정확도 계산."""
    # print(f'정확도 계산')
    right = np.argmax(preds, axis=1).astype(int) == np.concatenate(y).astype(int)
    accuracy = np.sum(right) / len(preds)
    return accuracy


def plot_AUC(test_dataset, test_preds, test_AUC, savepath="AUC.png"):
    """Validation set에 대한 AUC를 Plot으로 그린다."""
    precision, recall, _ = precision_recall_curve(test_dataset.data[:, :1], test_preds[:, 1])
    fpr, tpr, _ = roc_curve(test_dataset.data[:, :1], test_preds[:, 1])
    plt.figure()

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    f.suptitle("AUC %.4f" % test_AUC)
    f.set_size_inches((8, 4))

    axes[0].fill_between(recall, precision, step="post", alpha=0.2, color="b")
    axes[0].set_title("Recall-Precision Curve")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("ROC curve")
    plt.show()
    plt.savefig(savepath)
    print(savepath)

def plot_AUC_multi_class(test_dataset, test_preds, test_AUC, savepath="AUC.png"):
    """Validation set에 대한 AUC를 Plot으로 그린다."""
    
    test_label = test_dataset.data[:, :1]
    n_class = int(np.max(test_label)+1)
    test_label_onehot = np.array([np.eye(n_class, dtype=np.int_)[int(label)] for label in test_label])

    fpr = dict()
    tpr = dict()
    precision = dict()
    recall = dict()
    roc_auc = dict()
    from sklearn.metrics import auc
    for i in range(n_class):
        fpr[i], tpr[i], _ = roc_curve(test_label_onehot[:, i], test_preds[:, i])
        precision[i], recall[i], _ = precision_recall_curve(test_label_onehot[:, i], test_preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(test_label_onehot.ravel(), test_preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_class)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    from scipy import interp
    for i in range(n_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    f.suptitle("AUC %.4f" % test_AUC)
    f.set_size_inches((16, 8))

    lw=2

    axes[1].plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    axes[1].plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    from itertools import cycle
    colors = cycle(['red', 'aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_class), colors):
        axes[1].plot(fpr[i], tpr[i], color=color, lw=lw, label='ROC curve of class {0} (area = {1:0.2f})'
            ''.format(i, roc_auc[i]))
        axes[0].fill_between(recall[i], precision[i], step="post", alpha=0.2, color=color, label='class {0}'''.format(i))
    

    axes[1].plot([0, 1], [0, 1], 'k--', lw=lw)
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title("ROC curve")
    axes[1].legend(loc="lower right")
    axes[0].legend(loc="lower right")
    axes[0].set_title("Recall-Precision Curve")

    """
    plt.figure()

    f, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    f.suptitle("AUC %.4f" % test_AUC)
    f.set_size_inches((8, 4))

    axes[0].fill_between(recall, precision, step="post", alpha=0.2, color="b")
    axes[0].set_title("Recall-Precision Curve")

    axes[1].plot(fpr, tpr)
    axes[1].plot([0, 1], [0, 1], linestyle="--")
    axes[1].set_title("ROC curve")
    """
    plt.show()
    plt.savefig(savepath)
    print(savepath)

    return roc_auc


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
    plt.show()
    plt.savefig(savepath)
    print(savepath)
