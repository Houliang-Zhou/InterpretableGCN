import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm
import pdb
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import statistics
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import StepLR

from data import *
from kernel.sgcn import *
from Imbalanced import *
from utils_graph import *
from kernel.braingnn import Network
from utils import *

def cross_validation_with_val_set(args,dataset,target,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      gcn_num_layers,
                                      gcn_hidden,
                                      weight_decay,
                                      device,
                                      logger=None,
                                      result_path=None,
                                      pre_transform=None,
                                      result_file_name=None):
    test_data = []
    score_result = []
    test_losses, accs, durations = [], [], []
    test_mse = []
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, target, folds))):
        print("CV fold " + str(count))
        count += 1

        # train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]
        test_dataset = dataset[test_idx]
        train_y = target[train_idx]
        val_y = target[val_idx]
        test_y = target[test_idx]

        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))

        train_adj = build_adj_graph(train_dataset, topk_ratio=args.thredgraph)
        val_adj = build_adj_graph(val_dataset, topk_ratio=args.thredgraph)
        test_adj = build_adj_graph(test_dataset, topk_ratio=args.thredgraph)

        model = Network(args.max_len, 0.5, 2, R=100, topk_ratio=args.topk_ratio, device=device)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
        schedulerLR = StepLR(optimizer, step_size=80, gamma=0.1) #80

        train_dataset = SignalDataset(train_dataset, train_y, None, train_adj)
        val_dataset = SignalDataset(val_dataset, val_y, None, val_adj)
        test_dataset = SignalDataset(test_dataset, test_y, None, test_adj)

        if args.isUseSampler:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(
                train_dataset))  # True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

        score_result_epoch = []
        t_start = time.perf_counter()
        best_test_mse = np.inf
        best_test_loss = np.inf
        best_test_acc = np.inf
        best_test_auc = np.inf
        best_test_sen = np.inf
        best_test_spe = np.inf
        best_test_epoch = np.inf
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            message_train, kl_coef = train(epoch, model, optimizer, train_loader, device, args, scheduler)
            message_val, loss_val, loss_ce_val, val_classification_result = eval_loss(epoch, model, val_loader,
                                                                                             device, args)
            message_test, loss_test, loss_ce_test, classification_result = eval_loss(epoch, model, test_loader,
                                                                                             device, args)
            schedulerLR.step()
            logger(message_train)
            logger(message_test)

            acc = classification_result["acc"]
            auc = classification_result["auc"]
            sensitivity = classification_result["sensitivity"]
            specificity = classification_result["specificity"]
            accs.append(acc)
            test_losses.append(loss_test)

            if best_test_loss > loss_val:
                best_test_loss = loss_val
                message_best = 'Fold {:02d} Epoch {:04d} [Test classification BEST result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}'.format(
                    fold, epoch, acc, auc, sensitivity, specificity)
                logger(message_best)
                best_test_epoch = epoch
                best_test_acc = acc
                best_test_auc = auc
                best_test_sen = sensitivity
                best_test_spe = specificity
                model_path = os.path.join(args.res_dir, 'sgcn_ode_dict_contrastID%d_fold_%d_trainlen%d_maxlen%d.pt' % (
                    args.disease_id, fold, args.training_len, args.max_len))
                torch.save(model.state_dict(), model_path)

            message_best = 'Fold {:02d} Epoch {:04d} [Test classification result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}'.format(
                fold, epoch, acc, auc, sensitivity, specificity)
            logger(message_best)
            message_best = 'Fold {:02d} Epoch {:04d} [Test classification Current BEST result] | BestEpoch {:04d} | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}'.format(
                fold, epoch, best_test_epoch, best_test_acc, best_test_auc, best_test_sen, best_test_spe)
            logger(message_best)

            score_result_epoch.append([acc, auc, sensitivity, specificity])

        score_result.append(score_result_epoch)
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    # log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
    #        'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
    #     loss.mean().item(),
    #     acc_max.item(),
    #     acc[:, argmax].std().item(),
    #     acc_final.item(),
    #     acc[:, -1].std().item(),
    #     duration.mean().item()
    # )
    # if logger is not None:
    #     logger(log)

    score_result = np.asarray(score_result)
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def cross_validation_without_val_set( args,dataset,target,
                                      folds,
                                      epochs,
                                      batch_size,
                                      lr,
                                      lr_decay_factor,
                                      lr_decay_step_size,
                                      gcn_num_layers,
                                      gcn_hidden,
                                      weight_decay,
                                      device,
                                      logger=None,
                                      result_path=None,
                                      pre_transform=None,
                                      result_file_name=None):
    test_data = []
    score_result = []
    test_losses, accs, durations = [], [], []
    test_mse=[]
    count = 1
    for fold, (train_idx, test_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, target, folds))):
        print("CV fold " + str(count))
        count += 1

        train_idx = torch.cat([train_idx, val_idx], 0)  # combine train and val
        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        train_y = target[train_idx]
        test_y = target[test_idx]

        train_adj = build_adj_graph(train_dataset, topk_ratio=args.thredgraph)
        test_adj = build_adj_graph(test_dataset, topk_ratio=args.thredgraph)

        model = Network(args.max_len, 0.5, 2, R=100, topk_ratio=args.topk_ratio, device=device)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
        schedulerLR = StepLR(optimizer, step_size=20, gamma=0.5)

        train_dataset = SignalDataset(train_dataset, train_y, None, train_adj)
        test_dataset = SignalDataset(test_dataset, test_y, None, test_adj)

        if args.isUseSampler:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=ImbalancedDatasetSampler(
                train_dataset))  # True  False, sampler=ImbalancedDatasetSampler(train_dataset)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        score_result_epoch = []
        t_start = time.perf_counter()
        best_test_mse = np.inf
        best_test_loss = np.inf
        pbar = tqdm(range(1, epochs + 1), ncols=70)
        for epoch in pbar:
            message_train = train(epoch, model, optimizer, train_loader, device, args, scheduler)
            message_test, loss_test, loss_ce_test, classification_result = eval_loss(epoch, model, test_loader, device, args)
            schedulerLR.step()
            if message_train is not None:
                logger(message_train)
                logger(message_test)

            acc = classification_result["acc"]
            auc = classification_result["auc"]
            sensitivity = classification_result["sensitivity"]
            specificity = classification_result["specificity"]
            accs.append(acc)
            test_losses.append(loss_test)

            message_best = 'Fold {:02d} Epoch {:04d} [Test classification result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}|'.format(fold,epoch, acc, auc, sensitivity, specificity)
            if message_best is not None:
                logger(message_best)

            if best_test_loss>loss_test:
                best_test_loss = loss_test
                message_best = 'Fold {:02d} Epoch {:04d} [Test classification BEST result] | acc {:.6f}| auc {:.6f}| sensitivity {:.6f}| specificity {:.6f}|'.format(
                    fold,epoch, acc, auc, sensitivity, specificity)
                if message_best is not None:
                    logger(message_best)
                model_path = os.path.join(args.res_dir, 'sgcn_ode_dict_contrastID%d_fold_%d_trainlen%d_maxlen%d.pt' % (
                args.disease_id, fold, args.training_len, args.max_len))
                torch.save(model.state_dict(), model_path)

            score_result_epoch.append([acc,auc,sensitivity,specificity])

        score_result.append(score_result_epoch)
        t_end = time.perf_counter()
        durations.append(t_end - t_start)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    # log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
    #        'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
    #     loss.mean().item(),
    #     acc_max.item(),
    #     acc[:, argmax].std().item(),
    #     acc_final.item(),
    #     acc[:, -1].std().item(),
    #     duration.mean().item()
    # )
    # if logger is not None:
    #     logger(log)

    score_result = np.asarray(score_result)
    if result_path is not None:
        with open(result_path, 'wb') as f:
            np.save(f, score_result)

    return loss.mean().item(), acc_max.item(), acc[:, argmax].std().item()


def k_fold(dataset, target, folds):
    skf = StratifiedKFold(folds, shuffle=True, random_state=1000)

    test_indices, train_indices = [], []
    # tmp_label = [dataset[index].y.item() for index in range(len(dataset))]
    for _, idx in skf.split(torch.zeros(len(dataset)), np.reshape(target, -1)):  #
        test_indices.append(torch.from_numpy(idx).long())

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        # try:
        #     train_mask[test_indices[i]] = 0
        # except:
        #     a=1
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio, EPS = 1e-10):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s, device=None):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res


def compute_all_losses(model, input_data, ori_dada, target=None, adj=None, isOnlyTestReconstruct = False, args=None, rois = 100):
    recons_signal = input_data
    if args.build_graph_bycorr:
        input_x, edge_index, edge_weight = model.build_graph_byadj(recons_signal, adj)
    else:
        input_x, edge_index, edge_weight = model.build_graph(recons_signal)

    pos = F.one_hot(torch.arange(0, input_x.shape[0]) % model.R).float().to(model.device)
    B, N, T = recons_signal.shape
    batch = model.build_batch_num(B, N)
    out_softmax, w1, w2, s1, s2 = model(input_x, edge_index, batch, edge_weight, pos)

    loss_ce = F.nll_loss(out_softmax, target.view(-1))
    loss_p1 = (torch.norm(w1, p=2) - 1) ** 2
    loss_p2 = (torch.norm(w2, p=2) - 1) ** 2
    loss_tpk1 = topk_loss(s1, args.ratio)
    loss_tpk2 = topk_loss(s2, args.ratio)
    loss_consist = 0
    for c in range(args.nclass):
        loss_consist += consist_loss(s1[target.view(-1) == c], device=args.device)

    loss = args.lamb0 * loss_ce + args.lamb1 * loss_p1 + args.lamb2 * loss_p2 \
           + args.lamb3 * loss_tpk1 + args.lamb4 * loss_tpk2 + args.lamb5 * loss_consist
    pred_label = out_softmax.max(1)[1]
    pred_label = pred_label.view(-1)
    target = target.view(-1)

    results = {}
    results["loss"] = loss
    results["loss_tpk1"] = loss_tpk1
    results["loss_tpk2"] = loss_tpk2
    results["loss_ce"] = loss_ce
    results["loss_consist"] = loss_consist
    results["pred_label"] = pred_label.cpu().detach().numpy().tolist()
    results["target"] = target.cpu().detach().numpy().tolist()
    results["out_scores"] = out_softmax[:, 1]
    return results


def train(epo, model, optimizer, loader, device, args, scheduler, kl_coef = 1., wait_until_kl_inc = 10, fold=0):
    model.train()
    loss_tpk1_list = []
    loss_tpk2_list = []
    loss_ce_list = []
    loss_consist_list = []
    loss_list = []
    true_label = []
    pred_label = []
    itr=0
    for data, target, adj in loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        adj = adj.to(device)
        input_data = data
        if itr < wait_until_kl_inc:
            kl_coef = 0.
        else:
            kl_coef = (1 - 0.99 ** (itr - wait_until_kl_inc))
        results = compute_all_losses(model, input_data, data, target, adj, args=args)
        loss = results["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss_tpk1, loss_tpk2, loss_ce, loss_consist = results["loss_tpk1"].detach().data.item(), results["loss_tpk2"].detach().data.item(), \
                                                   results["loss_ce"].detach().data.item(), results["loss_consist"].detach().data.item()
        loss_list.append(loss.detach().item())
        loss_tpk1_list.append(loss_tpk1)
        loss_tpk2_list.append(loss_tpk2)
        loss_ce_list.append(loss_ce)
        loss_consist_list.append(loss_consist)
        itr+=1

    # scheduler.step()
    message_train = 'Fold {:02d} Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f} | loss_topk1 {:.6f}| loss_topk2 {:.6f}| loss_consist {:.6f}'.format(
        fold,epo,
        np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_tpk1_list), np.mean(loss_tpk2_list), np.mean(loss_consist_list))

    return message_train, kl_coef


def eval_loss(epo, model, loader, device, args, kl_coef=1.0, num_classes=2, fold=0):
    model.eval()
    loss_tpk1_list = []
    loss_tpk2_list = []
    loss_ce_list = []
    loss_consist_list = []
    loss_list = []
    true_label = []
    pred_label = []
    all_scores = []
    loss = 0
    for data, target, adj in loader:
        data = data.to(device)
        target = target.to(device)
        adj = adj.to(device)
        input_data = data
        with torch.no_grad():
            test_res = compute_all_losses(model, input_data, data, target, adj, args=args)
            loss = test_res["loss"]
            loss_tpk1, loss_tpk2, loss_ce, loss_consist = test_res["loss_tpk1"].detach().data.item(), test_res[
                "loss_tpk2"].detach().data.item(), test_res["loss_ce"].detach().data.item(), test_res[ "loss_consist"].detach().data.item()
            loss_list.append(loss.detach().item())
            loss_tpk1_list.append(loss_tpk1)
            loss_tpk2_list.append(loss_tpk2)
            loss_ce_list.append(loss_ce)
            loss_consist_list.append(loss_consist)

            all_scores.append(test_res["out_scores"].cpu().detach())
            true_label = true_label + [per_label for per_label in test_res["target"]]
            pred_label = pred_label + [per_label for per_label in test_res["pred_label"]]

    message_test = 'Fold {:02d} Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f} | loss_topk1 {:.6f}| loss_topk2 {:.6f}| loss_consist {:.6f}'.format(
        fold,epo,
        np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_tpk1_list), np.mean(loss_tpk2_list), np.mean(loss_consist_list))

    all_scores = torch.cat(all_scores).cpu().numpy()
    auc = 0
    try:
        if num_classes < 3:
            fpr, tpr, _ = metrics.roc_curve(np.asarray(true_label), all_scores, pos_label=1)
            auc = metrics.auc(fpr, tpr)
    except:
        auc = 0

    true_label = np.asarray(true_label)
    pred_label = np.asarray(pred_label)
    N = true_label.shape[0]
    test_f1 = f1_score(true_label, pred_label, average='weighted')
    acc = (true_label == pred_label).sum() / N
    if num_classes < 3:
        cm = confusion_matrix(true_label, pred_label)
        TN, FP, FN, TP = cm.ravel()
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
    else:
        sensitivity = 0
        specificity = 0

    classification_result = {}
    classification_result["acc"] = acc
    classification_result["auc"] = auc
    classification_result["sensitivity"] = sensitivity
    classification_result["specificity"] = specificity

    return message_test, np.mean(loss_list), np.mean(loss_ce_list), classification_result
