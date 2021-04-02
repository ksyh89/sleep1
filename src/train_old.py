import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data

from dataset import Dataset
from model import get_classifier_model
from settings import TrainInformation, TrainResult
from utils import train_utils

plt.interactive(True)

prev_plot = 0


def set_optimizer(optimizer_method, model, init_lr, weight_decay, momentum=None):
    """Optimizer 설정."""
    if optimizer_method == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=init_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    elif optimizer_method == "Adadelta":
        optimizer = torch.optim.Adadelta(
            model.parameters(), lr=init_lr, weight_decay=weight_decay
        )
    else:
        raise Exception("Unknown Optimizer {}".format(optimizer_method))
    return optimizer


def print_metrics(model, train_dataset, test_dataset, train_result):
    model.train(False)
    test_preds = train_utils.get_preds(test_dataset.data[:, 1:], model)
    test_AUC = train_utils.compute_AUC(test_dataset.data[:, :1], test_preds)
    test_accuracy = train_utils.compute_accuracy(test_dataset.data[:, :1], test_preds)

    train_preds = train_utils.get_preds(train_dataset.data[:1000, 1:], model)
    train_AUC = train_utils.compute_AUC(train_dataset.data[:1000, :1], train_preds)

    train_accuracy = train_utils.compute_accuracy(
        train_dataset.data[:1000, :1], train_preds
    )

    train_result.test_AUC_list.append("%.04f" % test_AUC)
    train_result.test_accuracy_list.append("%.04f" % test_accuracy)

    return train_AUC, test_AUC, train_accuracy, test_accuracy, test_preds


def compute_contributing_variables(model, test_dataset):
    print("Evaluating contributing variables")
    model.train(False)
    variable_by_column = np.load("../datasets/medical_data_6_no_space_columnnames.npy")
    #variable_by_column = np.array([v.replace("HE_ast", "HE_alt") for v in variable_by_column])
    assert variable_by_column.shape[0] == test_dataset.data.shape[1] - 1
    variables = np.unique(variable_by_column)
    AUCs = []
    print("Computing variable contributions")
    print(variables)
    for variable in variables:
        corresponding_indices = (variable_by_column == variable)
        #print("zeroing %s" % str(np.where(corresponding_indices)))
        val_data = test_dataset.data[:, 1:].copy()
        val_data[:, corresponding_indices] = 0.0
        #print((val_data[:, :17] ** 2).mean())
        #val_data = val_data * len(variables) / (len(variables) - 1)
        preds = train_utils.get_preds(val_data, model)
        target = test_dataset.data[:, :1]
        test_AUC = train_utils.compute_AUC(target, preds)
        print("%s %f" % (variable, test_AUC))
        AUCs.append(test_AUC)

    sorting_indices = np.argsort(AUCs)
    sorted_variables = [variables[i] for i in sorting_indices]
    sorted_AUCs = [AUCs[i] for i in sorting_indices]

    sorted_pairs = [(v, auc) for (v, auc) in zip(sorted_variables, sorted_AUCs)]
    for i, (v, auc) in enumerate(sorted_pairs[:20]):
        print("%03d: %s %f" % (i, v, auc))
        
    return [(v, auc) for (v, auc) in zip(variables, AUCs)]


def train_step(
        exp_name,
        ep,
        model,
        train_dataset,
        test_dataset,
        optimizer,
        init_lr,
        lr_decay,
        data_loader,
        bce_loss,
        train_result: TrainResult,
):
    global prev_plot
    model.train(True)
    for _, (X, y) in enumerate(data_loader):
        optimizer.zero_grad()
        pred_out = model(X.cuda()).view(X.shape[0])
        loss = bce_loss(pred_out, y.cuda())
        loss.backward()
        avg_loss = train_result.avg_loss * 0.98 + loss.detach().cpu().numpy() * 0.02
        optimizer.step()
        train_result.total_iter += len(y)
        if train_result.total_iter % 10000 == 0:
            print(
                "Loss Iter %05d: %.4f\r" % (train_result.total_iter, avg_loss), end=""
            )
            train_result.loss_list.append(
                (train_result.total_iter, "{:.4f}".format(avg_loss))
            )
    print("")

    lr = init_lr * (lr_decay ** ep)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Learning rate = %f" % lr)

    train_AUC, test_AUC, train_accuracy, test_accuracy, test_preds = print_metrics(model,
                                                                                   train_dataset,
                                                                                   test_dataset,
                                                                                   train_result)
    savedir = "checkpoints/%s" % exp_name
    os.makedirs(savedir, exist_ok=True)
    split = train_dataset.split
    savepath = "%s/epoch_%04d_fold_%02d.pt" % (savedir, ep, split)
    torch.save(model, savepath)

    if train_result.best_test_AUC < test_AUC:
        train_result.best_test_AUC = test_AUC
        train_result.best_test_epoch = ep
        if ep - prev_plot > 10:
            # 너무 자주 찍지 말고 한번 plot 찍고 epoch 10번 이상인 경우에만 찍는다.
            prev_plot = ep
            #train_utils.plot_AUC(test_dataset, test_preds, test_AUC)
        #contributing_variables = compute_contributing_variables(model, test_dataset)

    print(
        "Epoch %03d: test_AUC: %.4f (best: %.4f epoch: %d), train_AUC: %.4f"
        % (
            ep,
            test_AUC,
            train_result.best_test_AUC,
            train_result.best_test_epoch,
            train_AUC,
        )
    )
    print(
        "            test_accuracy {:.4f}, train_accuracy {:.4f}".format(
            test_accuracy, train_accuracy,
        )
    )


def train_logisticregressoin(info: TrainInformation, split, fold):
    """주어진 split에 대한 학습과 테스트를 진행한다."""
    bs = info.BS
    init_lr = info.INIT_LR
    lr_decay = info.LR_DECAY
    momentum = info.MOMENTUM
    weight_decay = info.WEIGHT_DECAY
    optimizer_method = info.OPTIMIZER_METHOD
    epoch = info.EPOCH
    nchs = info.NCHS
    filename = info.FILENAME
    model_name = info.MODEL_NAME
    exp_name = info.NAME

    print("Using File {}".format(filename))

    train_dataset = Dataset(split=split, fold=fold, phase="train", filename=filename, use_data_dropout=info.USE_DATA_DROPOUT)
    #val_dataset = Dataset(split=split, fold=fold, phase="val", filename=filename)
    test_dataset = Dataset(split=split, fold=fold, phase="test", filename=filename, use_data_dropout=False)

    import sklearn.linear_model

    regressor = sklearn.linear_model.LogisticRegression()
    regressor.fit(train_dataset.train_data[:, 1:], test_dataset.train_data[:, :1])
    preds = regressor.predict_proba(test_dataset.data[:, 1:])[:, 1]
    auc = train_utils.compute_AUC(test_dataset.data[:, :1], preds)
    print(auc)
    savepath = "checkpoints/logistic_regression/split_%02d.png" % split
    os.makedirs(os.path.dirname(savepath), exist_ok=True)
    #train_utils.plot_AUC_v2(preds, test_dataset.data[:, :1], savepath=savepath)

    model = get_classifier_model(model_name, train_dataset.feature_size, nchs, info.ACTIVATION)
    savedir = "checkpoints/%s" % exp_name
    best_test_epoch = 25
    loadpath = "%s/epoch_%04d_fold_%02d.pt" % (savedir, best_test_epoch, train_dataset.split)
    #model.load_state_dict(torch.load(savepath))
    model = torch.load(loadpath)
    model.eval()

    test_preds = train_utils.get_preds(test_dataset.data[:, 1:], model)
    train_utils.plot_AUC_v2([('Deep Neural Network', test_preds), ('Logistic Regression', preds)], test_dataset.data[:, :1], savepath=savepath)
    

def train(info: TrainInformation, split, fold):
    """주어진 split에 대한 학습과 테스트를 진행한다."""
    bs = info.BS
    init_lr = info.INIT_LR
    lr_decay = info.LR_DECAY
    momentum = info.MOMENTUM
    weight_decay = info.WEIGHT_DECAY
    optimizer_method = info.OPTIMIZER_METHOD
    epoch = info.EPOCH
    nchs = info.NCHS
    filename = info.FILENAME
    model_name = info.MODEL_NAME
    exp_name = info.NAME

    print("Using File {}".format(filename))

    train_dataset = Dataset(split=split, fold=fold, phase="train", filename=filename, use_data_dropout=info.USE_DATA_DROPOUT)
    #val_dataset = Dataset(split=split, fold=fold, phase="val", filename=filename)
    test_dataset = Dataset(split=split, fold=fold, phase="test", filename=filename, use_data_dropout=False)

    model = get_classifier_model(model_name, train_dataset.feature_size, nchs, info.ACTIVATION)
    

    print(model)

    # Optimizer 설정
    optimizer = set_optimizer(
        optimizer_method, model, init_lr, weight_decay, momentum=momentum
    )

    data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bs, shuffle=True, num_workers=0, drop_last=True
    )

    bce_loss = torch.nn.BCEWithLogitsLoss().cuda()
    train_result = TrainResult()
    train_result.set_sizes(
        len(train_dataset.data), 0, len(test_dataset.data)
    )

    for ep in range(epoch):
        global prev_plot
        prev_plot = 0
        train_step(
            exp_name,
            ep,
            model,
            train_dataset,
            test_dataset,
            optimizer,
            init_lr,
            lr_decay,
            data_loader,
            bce_loss,
            train_result,
        )

    savedir = "checkpoints/%s" % exp_name
    best_test_epoch = 25
    savepath = "%s/epoch_%04d_fold_%02d.pt" % (savedir, best_test_epoch, train_dataset.split)
    #model.load_state_dict(torch.load(savepath))
    model = torch.load(savepath)
    model.eval()

    test_preds = train_utils.get_preds(test_dataset.data[:, 1:], model)
    test_AUC = train_utils.compute_AUC(test_dataset.data[:, :1], test_preds)
    train_utils.plot_AUC(test_dataset, test_preds, test_AUC, savepath=savepath.replace(".pt", "_AUC.png"))

    contributing_variables = compute_contributing_variables(model, test_dataset)
    with open(os.path.join(savedir, "contributing_variables_epoch_%04d_fold_%02d.txt" % (best_test_epoch, train_dataset.split)), "w") as f:
        for (v, auc) in contributing_variables:
            f.write("%s %f\n" % (v, auc))

    
    info.split_index = split
    info.result_dict = train_result
    info.save_result()
    return train_result


def run(filename):
    """실험할 세팅을 불러오고, 그에 따라서 실험을 수행한다."""
    info = TrainInformation(filename)
    np.random.seed(info.SEED)
    torch.manual_seed(info.SEED)
    fold = info.FOLD

    test_AUCs_by_split = []
    for split in range(fold):
        
        #if split % 3 > 0:
        #    print("Skipping split %d" % split)
        #    continue
        if False:
            train_logisticregressoin(info, split, fold)
            continue
        result = train(info, split, fold)
        test_AUCs = [float(auc) for auc in result.test_AUC_list]
        test_AUCs_by_split.append(test_AUCs)

    with open("result.txt", "a") as f:
        test_AUCs_by_split = np.array(test_AUCs_by_split)
        test_AUCs_by_epoch = test_AUCs_by_split.mean(axis=0)
        best_test_epoch = np.argmax(test_AUCs_by_epoch)
        best_test_AUC = test_AUCs_by_epoch[best_test_epoch]
        #f.write(str(info) + "/n")
        f.write("Name: %s\n" % info.NAME)
        f.write("average test AUC: %f %d\n" % (best_test_AUC, best_test_epoch))


if __name__ == "__main__":
    # train 함수를 직접 호출했을 때 실행.
    data_path = "../datasets/medical_data_6_no_space.csv"
    run(data_path)
