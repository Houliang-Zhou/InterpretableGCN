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
from lib.likelihood_eval import *
from Imbalanced import *
from utils_graph import *

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
        # 5, 64, 16

        train_adj = build_adj_graph(train_dataset, topk_ratio=args.sgcn_thredgraph)
        val_adj = build_adj_graph(val_dataset, topk_ratio=args.sgcn_thredgraph)
        test_adj = build_adj_graph(test_dataset, topk_ratio=args.sgcn_thredgraph)

        model = SGCN_GCN(graphODE_Model=None, num_layers=gcn_num_layers, hidden=gcn_hidden, rois=100, num_features=args.max_len,
                         topk_ratio=args.sgcn_thredgraph, pooling=args.pooling, device=device)
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
            message_train = train(epoch, model, optimizer, train_loader, device, args, scheduler)
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

            if best_test_loss > loss_ce_val:
                best_test_loss = loss_ce_val
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

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
           'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    if logger is not None:
        logger(log)

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

        obsrv_std = 0.01
        obsrv_std = torch.Tensor([obsrv_std]).to(device)
        z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
        # 5, 64, 16

        train_adj = build_adj_graph(train_dataset, topk_ratio=args.sgcn_thredgraph)
        test_adj = build_adj_graph(test_dataset, topk_ratio=args.sgcn_thredgraph)

        model = SGCN_GCN(None, num_layers=gcn_num_layers, hidden=gcn_hidden, rois=100, num_features=args.max_len, topk_ratio=args.sgcn_thredgraph, pooling=args.pooling, device=device)
        model = model.to(device)
        # optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if args.optimizer == "AdamW":
            optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.l2)
        else:
            optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)
        # schedulerLR = StepLR(optimizer, step_size=50, gamma=0.5)

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
            #schedulerLR.step()
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

    # loss, duration = tensor(test_mse), tensor(durations)
    # loss = loss.view(folds, epochs)
    # acc_mean = loss.mean(0)
    # acc_max, argmax = acc_mean.min(dim=0)
    # acc_final = acc_mean[-1]
    #
    # log = ('Test Loss: {:.4f}, Test Min MSE: {:.3f}, ' +
    #       'Test Final MSE: {:.3f}, Duration: {:.3f}').format(
    #     loss.mean().item(),
    #     acc_max.item(),
    #     acc_final.item(),
    #     duration.mean().item()
    # )
    # if logger is not None:
    #     logger(log)

    loss, acc, duration = tensor(test_losses), tensor(accs), tensor(durations)
    loss, acc = loss.view(folds, epochs), acc.view(folds, epochs)
    acc_mean = acc.mean(0)
    acc_max, argmax = acc_mean.max(dim=0)
    acc_final = acc_mean[-1]

    log = ('Test Loss: {:.4f}, Test Max Accuracy: {:.3f} ± {:.3f}, ' +
           'Test Final Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}').format(
        loss.mean().item(),
        acc_max.item(),
        acc[:, argmax].std().item(),
        acc_final.item(),
        acc[:, -1].std().item(),
        duration.mean().item()
    )
    if logger is not None:
        logger(log)

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


def compute_all_losses(model, input_data, ori_dada, target=None, adj=None, args=None):
    # if args.iswoODE:
    #     recons_signal = input_data
    # else:
    #     recons_signal, info = model.graphode(input_data)
    # if args.isCatReconstructed:
    #     notzeros_value = (mask!=0).all(dim=1)
    #     recons_signal = recons_signal.detach()
    #     recons_signal = torch.cat((input_data, recons_signal[:,:,args.training_len:]),-1)
    #     recons_signal[notzeros_value] = ori_dada[notzeros_value]

    recons_signal = input_data
    # recons_signal = recons_signal.detach()
    if args.build_graph_bycorr:
        input_x, edge_index, edge_weight = model.build_graph_byadj(recons_signal, adj)
    else:
        input_x, edge_index, edge_weight = model.build_graph(recons_signal)
    out_softmax = model(recons_signal, input_x, edge_index, edge_weight)
    out_prob = model(recons_signal, input_x, edge_index, edge_weight, isExplain=True)

    loss_ce = F.nll_loss(out_softmax, target.view(-1))
    loss_mi = F.nll_loss(out_prob, target.view(-1))
    loss_prob = model.loss_probability(input_x, edge_index, edge_weight, args)
    loss = args.lamda_ce * loss_ce + args.lamda_prob * loss_prob + args.lamda_mi * loss_mi

    pred_label = out_softmax.max(1)[1]
    pred_label = pred_label.view(-1)
    target = target.view(-1)

    results = {}
    results["loss"] = loss
    results["loss_ce"] = loss_ce
    results["loss_prob"] = loss_prob
    results["loss_mi"] = loss_mi
    results["pred_label"] = pred_label.cpu().detach().numpy().tolist()
    results["target"] = target.cpu().detach().numpy().tolist()
    results["out_scores"] = out_softmax[:, 1]
    return results


def train(epo, model, optimizer, loader, device, args, scheduler, fold=0):
    model.train()
    loss_ce_list = []
    loss_prob_list = []
    loss_mi_list = []
    loss_list = []
    itr=0
    for data, target, adj in loader:
        optimizer.zero_grad()
        data = data.to(device)
        target = target.to(device)
        adj = adj.to(device)
        input_data = data

        results = compute_all_losses(model, input_data, data, target=target, adj=adj, args=args)
        loss = results["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        loss_ce, loss_prob, loss_mi = results["loss_ce"].detach().data.item(), results["loss_prob"].detach().data.item(), results["loss_mi"].detach().data.item()
        loss_list.append(loss.detach().item())
        loss_ce_list.append(loss_ce)
        loss_prob_list.append(loss_prob)
        loss_mi_list.append(loss_mi)
        itr+=1

    if args.isCosineAnnealingLR:
        scheduler.step()
    message_train = 'Fold {:02d} Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f}| loss_prob {:.6f}| loss_mi {:.6f}'.format(
        fold,epo,np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_prob_list), np.mean(loss_mi_list))

    return message_train


def eval_loss(epo, model, loader, device, args, num_classes=2, fold=0):
    model.eval()
    loss_ce_list = []
    loss_prob_list = []
    loss_mi_list = []
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
            test_res = compute_all_losses(model, input_data, data, target=target, adj=adj, args=args)
            loss = test_res["loss"]
            loss_ce, loss_prob, loss_mi = test_res["loss_ce"].detach().data.item(), test_res["loss_prob"].detach().data.item(), test_res[
                                                        "loss_mi"].detach().data.item()
            loss_list.append(loss.detach().item())
            loss_ce_list.append(loss_ce)
            loss_prob_list.append(loss_prob)
            loss_mi_list.append(loss_mi)
            all_scores.append(test_res["out_scores"].cpu().detach())
            true_label = true_label + [per_label for per_label in test_res["target"]]
            pred_label = pred_label + [per_label for per_label in test_res["pred_label"]]

    message_test = 'Fold {:02d} Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | loss_ce {:.6f}| loss_prob {:.6f}| loss_mi {:.6f}'.format(
        fold,epo,np.mean(loss_list), np.mean(loss_ce_list), np.mean(loss_prob_list), np.mean(loss_mi_list))

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
