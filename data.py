import math
import os.path

import numpy
import pandas as pd
import sklearn
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import coo_matrix
import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset
# print(torch.__version__)
from torch_geometric.data import Data
from tqdm import tqdm
import scipy.io as sio
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas import read_csv
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=0, isUseDXLabel=False):
    print("select disease id: %d; isUseDXLabel: "%(disease_id), isUseDXLabel)
    if isUseDXLabel:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        len_raw_signal = len_raw_signal[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: ", normal_sum, abnormal_sum)
        return normalized_data, len_raw_signal, fmri_subid, dx_label
    else:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((label == 0) | (label == 4))[0]
        elif disease_id == 1:
            select_indices = np.where((label == 0) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((label == 4) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 3:
            select_indices = np.where((label == 0) |(label == 4) | (label == 2) | (label == 3) | (label == 1))[0]
        elif disease_id == 4:
            select_indices = np.where((label == 0) | (label == 1))[0]
        elif disease_id == 5:
            select_indices = np.where((label == 0) | (label == 2))[0]
        elif disease_id == 6:
            select_indices = np.where((label == 0) | (label == 3))[0]
        else:
            select_indices = np.where((label < 5))[0]

        normalized_data = normalized_data[select_indices]
        len_raw_signal = len_raw_signal[select_indices]
        fmri_subid = fmri_subid[select_indices]
        label = label[select_indices]

        if disease_id == 2:
            label[label != 4] = 0
            label[label == 4] = 1
        else:
            label[label > 0] = 1

        normal_sum = np.sum(label == 0)
        abnormal_sum = np.sum(label > 0)
        print("Num of normal and abnormal samples: ",normal_sum, abnormal_sum)
        return normalized_data, len_raw_signal, fmri_subid, label

def generate_Mask(len_raw_signal, maxlen = 200):
    mask = []
    for i in range(len(len_raw_signal)):
        len_one = len_raw_signal[i]
        result_one = [1]*len_one
        if len_one<maxlen:
            result_one += [0]*(maxlen-len_one)
        mask.append(result_one)
    mask = np.asarray(mask)
    return mask

def load_fmri(root, disease_id=0, isNormalize = True, isPadding=True, maxlen = 200,
              isOnlySelectFullSig = False, isSelfNormalize=False, isDrawSection=False, isUseDXLabel = False):
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id']
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label']
    label = sio.loadmat(path_label)
    label = label['label']
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)

    if isDrawSection:
        values, counts = np.unique(fmri_subid, return_counts=True)
        # matplotlib histogram
        plt.hist(counts, color='blue', edgecolor='black', bins=8)

        print(np.sum(counts==1))

        # seaborn histogram
        # sns.distplot(flights['arr_delay'], hist=True, kde=False,
        #              bins=int(180 / 5), color='blue',
        #              hist_kws={'edgecolor': 'black'})
        # Add labels
        # plt.title('Histogram of Arrival Delays')
        plt.xlabel('Number of sections')
        plt.ylabel('Count')
        plt.show()

    all_data = []
    len_raw_signal = []
    if isNormalize:
        if isSelfNormalize:
            normalized_data = []
            scaler = StandardScaler()
            for i in range(num_data):
                tmp_sig = fmri_signal[i][0]
                scaler.fit(tmp_sig)
                signal_sub = scaler.transform(tmp_sig)
                len_raw_signal.append(len(signal_sub))
                if isPadding and len(signal_sub) < maxlen:
                    signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                                        mode='constant')
                signal_sub = np.expand_dims(signal_sub, 0)
                normalized_data.append(signal_sub)
        else:
            for i in range(num_data):
                all_data.append(fmri_signal[i][0])
            all_data = np.concatenate(all_data)
            scaler = StandardScaler()
            scaler.fit(all_data)
            # print(all_data.shape)

            normalized_data = []
            for i in range(num_data):
                signal_sub = fmri_signal[i][0]
                signal_sub = scaler.transform(signal_sub)
                len_raw_signal.append(len(signal_sub))
                if isPadding and len(signal_sub)<maxlen:
                    signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen-len(signal_sub)), (0, 0)], mode='constant')
                signal_sub = np.expand_dims(signal_sub, 0)
                normalized_data.append(signal_sub)
        normalized_data = np.concatenate(normalized_data)
        normalized_data = np.transpose(normalized_data, (0, 2, 1))
        len_raw_signal = np.asarray(len_raw_signal)
        print("num of len 140: %d ;len 200: %d"%(num_data-np.sum(len_raw_signal==200), num_data-np.sum(len_raw_signal==140)))
        sig_mask = []
        if not isOnlySelectFullSig:
            normalized_data, len_raw_signal, fmri_subid, label = sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=disease_id, isUseDXLabel=isUseDXLabel)
            sig_mask = generate_Mask(len_raw_signal)
        else:
            normalized_data, len_raw_signal, fmri_subid, label = sample_bylabels(normalized_data, len_raw_signal, fmri_subid, label, dx_label, disease_id=-1, isUseDXLabel=isUseDXLabel)
            sig_mask = generate_Mask(len_raw_signal)
            index_select = len_raw_signal == maxlen
            normalized_data = normalized_data[index_select]
            len_raw_signal = len_raw_signal[index_select]
            sig_mask = sig_mask[index_select]
            fmri_subid = fmri_subid[index_select]
            label = label[index_select]
        print("num of sample selected: %d" % (normalized_data.shape[0]))
        # normalized_data = torch.from_numpy(normalized_data).float()
        # len_raw_signal = torch.from_numpy(len_raw_signal).float()
        return normalized_data, len_raw_signal, sig_mask, fmri_subid, label
    a=1
    return

def sub_signal(data, sub_len = 100):
    sub_data = data[:,:,:sub_len]
    return sub_data

def get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl):
    select_indices = np.where(index_bl & (dx_label>=0))[0]
    n = dx_label.shape[0]
    subid_wbl = fmri_subid[select_indices]
    dx_label_wbl = dx_label[select_indices]
    baseline_map={}
    for i in range(len(subid_wbl)):
        baseline_map[subid_wbl[i]] = dx_label_wbl[i]
    conversion_label = []
    for cursub in subid_wbl:
        index_cursub = fmri_subid==cursub
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label = dx_label[index_cursub]
        cur_isconver = 0
        cur_conver_label = baseline_map[cursub]
        max_timepoint = 0
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] > max_timepoint and cur_label[j] != baseline_map[cursub] and cur_label[j]>baseline_map[cursub] :
                cur_isconver = 1
                cur_conver_label = cur_label[j]
                max_timepoint = cur_timepoint[j]
        conversion_label.append([cur_isconver, baseline_map[cursub], cur_conver_label])
    conversion_label = np.asarray(conversion_label)
    print("num of convered subject: %d/%d"%(np.sum(conversion_label[:,0]), len(select_indices)))

def get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int):
    # select_indices = np.where(dx_label >= 0)[0]
    # sub_wlabel = fmri_subid[select_indices]
    n = len(fmri_subid)
    subid_baseline_map = {}
    isbaseline = np.zeros(n)
    for index_cursub in range(len(fmri_subid)):
        cursub = fmri_subid[index_cursub]
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label=dx_label[index_cursub]
        if cur_label>=0 and cur_timepoint>=0:
            if cursub not in subid_baseline_map or cur_timepoint<subid_baseline_map[cursub][1]:
                subid_baseline_map[cursub]=[index_cursub, cur_timepoint, cur_label]
    for key in subid_baseline_map:
        isbaseline[subid_baseline_map[key][0]]=1
    # print(len(subid_baseline_map.keys()))
    isbaseline = isbaseline.astype(bool)
    print("num of subject who has bl or starting signal with dx label: %d/%d"%(np.sum(isbaseline), len(isbaseline)))

    conversion_label = -np.ones(n)
    num_one_tp = 0
    check_error = []
    for cursub in subid_baseline_map:
        index_cursub = fmri_subid == cursub
        cur_timepoint = timepoint_id_int[index_cursub]
        cur_label = dx_label[index_cursub]
        cur_isconver = -1
        bl_index = subid_baseline_map[cursub][0]
        bl_tp = subid_baseline_map[cursub][1]
        bl_label = subid_baseline_map[cursub][2]
        max_timepoint = bl_tp
        cur_conver_label = bl_label
        if len(cur_timepoint)<=1:
            num_one_tp+=1
            check_error.append(cur_isconver)
            continue
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] != bl_tp and cur_label[j] >= 0:
                cur_isconver=0
                break
        for j in range(len(cur_timepoint)):
            if cur_timepoint[j] > max_timepoint and cur_label[j] > bl_label and cur_label[j] >=0:
                cur_isconver = cur_label[j]*2 - bl_label
                cur_conver_label = cur_label[j]
                max_timepoint = cur_timepoint[j]
        conversion_label[bl_index] = cur_isconver
        check_error.append(cur_isconver)
        # conversion_label.append([cur_isconver, baseline_map[cursub], cur_conver_label])
    # conversion_label = np.asarray(conversion_label)
    print("num of subject with only one timepoint", num_one_tp)
    print("num of convered subject: %d/%d; HC-MCI:%d; MCI-AD:%d; HC-AD:%d; " % (np.sum(conversion_label>=1), np.sum(conversion_label>=0), np.sum(conversion_label==2), np.sum(conversion_label==3), np.sum(conversion_label==4)))
    check_error_values, check_error_counts = np.unique(np.asarray(check_error), return_counts=True)
    print("check_error for convered subject: ", check_error_values, check_error_counts)

    return isbaseline, conversion_label

def load_fmri_AllADNI(root, disease_id=0, isNormalize = True, isPadding=True, maxlen = 197,
                      isOnlySelectFullSig = False, isSelfNormalize=False, isDrawSection=False, isUseDXLabel = False,
                      isSelect4Reconstruct = False, UseReconstructSignal = True):
    root = os.path.join(root, 'ADNI2-3')
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    path_index_bl = os.path.join(root, 'index_bl.mat')
    path_index_noradni3_or_norsmalllen = os.path.join(root, 'index_noradni3_or_norsmalllen.mat')
    path_timepoint_id_int = os.path.join(root, 'timepoint_id_int.mat')
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id'].reshape(-1)
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label'].reshape(-1)
    index_bl = sio.loadmat(path_index_bl)
    index_bl = index_bl['index_bl'].reshape(-1).astype(bool)
    index_noradni3_or_norsmalllen = sio.loadmat(path_index_noradni3_or_norsmalllen)
    index_noradni3_or_norsmalllen = index_noradni3_or_norsmalllen['index_noradni3_or_norsmalllen']
    timepoint_id_int = sio.loadmat(path_timepoint_id_int)
    timepoint_id_int = timepoint_id_int['timepoint_id_int'].reshape(-1)
    label = sio.loadmat(path_label)
    label = label['label'].reshape(-1)
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)
    fmri_subid_values, fmri_subid_counts = np.unique(fmri_subid, return_counts=True)
    print("num of subject: ", len(fmri_subid_counts))
    print("num of subject with bl signal: ", np.sum(index_bl))
    print("num of subject with timepoint >=0 : ", np.sum(timepoint_id_int>=0))
    # get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl)
    isbaseline, conversion_label = get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int)

    len_raw_signal = []
    for i in range(num_data):
        signal_sub = fmri_signal[i][0]
        if isbaseline[i]:
            len_raw_signal.append(len(signal_sub))
    len_raw_signal = np.asarray(len_raw_signal)
    num_bl_considerWOBLlabel = len(len_raw_signal)
    print("Baseline sub: num of small len: %d/%d ;len %d: %d/%d" % (
        num_bl_considerWOBLlabel - np.sum(len_raw_signal >= maxlen), num_bl_considerWOBLlabel, maxlen, num_bl_considerWOBLlabel - np.sum(len_raw_signal < maxlen),
        num_bl_considerWOBLlabel))

    len_raw_signal = []
    normalized_data = []
    if isNormalize and not UseReconstructSignal:
        normalized_data, len_raw_signal = normalize_sigdata(num_data, fmri_signal, maxlen=maxlen, isSelfNormalize=isSelfNormalize, isPadding=isPadding,UseReconstructSignal=UseReconstructSignal)
    else:
        for i in range(num_data):
            signal_sub = fmri_signal[i][0]
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant')
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
        normalized_data = np.concatenate(normalized_data)
        normalized_data = np.transpose(normalized_data, (0, 2, 1))
        len_raw_signal = np.asarray(len_raw_signal)
        print("signal min length:%d" % (np.min(len_raw_signal)))
        print("num of small len: %d/%d ;len %d: %d/%d" % (
            num_data - np.sum(len_raw_signal >= maxlen), num_data, maxlen, num_data - np.sum(len_raw_signal < maxlen),
            num_data))

    if UseReconstructSignal:
        print("raw_data: num of small value:", np.sum(normalized_data < 3.0))
        index_small_len = len_raw_signal < maxlen
        select_normalized_data = normalized_data[index_small_len]
        select_dx_label = dx_label[index_small_len]
        select_timepoint_id_int = timepoint_id_int[index_small_len]
        select_fmri_subid = fmri_subid[index_small_len]
        # print("isSelect4Reconstruct: normalized_data shape", select_normalized_data.shape)
        all_recons_sig = chooseReconstrctDataBySDE(raw_data=normalized_data, maxlen=maxlen, index_small_len=index_small_len, select_normalized_data=select_normalized_data, select_fmri_subid=select_fmri_subid)
        normalized_recons_sig, _ = normalize_sigdata(num_data, all_recons_sig, maxlen=maxlen,
                                                            isSelfNormalize=False, isPadding=False, UseReconstructSignal=UseReconstructSignal)
        with open(os.path.join(root, "normalized_recons_sig_bySDE.npy"), 'wb') as f:
            np.save(f, normalized_recons_sig)
        return normalized_recons_sig, dx_label, timepoint_id_int

    if isSelect4Reconstruct:
        index_small_len = len_raw_signal < maxlen
        select_normalized_data = normalized_data[index_small_len]
        select_dx_label = dx_label[index_small_len]
        select_timepoint_id_int = timepoint_id_int[index_small_len]
        select_fmri_subid = fmri_subid[index_small_len]
        print("isSelect4Reconstruct: normalized_data shape",select_normalized_data.shape)
        return select_normalized_data, select_dx_label, select_timepoint_id_int, select_fmri_subid

    return normalized_data, dx_label, timepoint_id_int

def load_recons_fmri_AllADNI(root, disease_id=0, isUseDXLabelOnly=True, isUseDXLabelwithBaselineOnly = False,
                               isTestConversion = False):
    root = os.path.join(root, 'ADNI2-3')
    path_signal = os.path.join(root, 'fmri_signal.mat')
    path_label = os.path.join(root, 'label.mat')
    path_dx_label = os.path.join(root, 'dx_label.mat')
    path_subid = os.path.join(root, 'subject_id.mat')
    path_index_bl = os.path.join(root, 'index_bl.mat')
    path_index_noradni3_or_norsmalllen = os.path.join(root, 'index_noradni3_or_norsmalllen.mat')
    path_timepoint_id_int = os.path.join(root, 'timepoint_id_int.mat')
    fmri_signal = sio.loadmat(path_signal)
    fmri_signal = fmri_signal['fmri_signal']
    fmri_subid = sio.loadmat(path_subid)
    fmri_subid = fmri_subid['subject_id'].reshape(-1)
    dx_label = sio.loadmat(path_dx_label)
    dx_label = dx_label['dx_label'].reshape(-1)
    index_bl = sio.loadmat(path_index_bl)
    index_bl = index_bl['index_bl'].reshape(-1).astype(bool)
    index_noradni3_or_norsmalllen = sio.loadmat(path_index_noradni3_or_norsmalllen)
    index_noradni3_or_norsmalllen = index_noradni3_or_norsmalllen['index_noradni3_or_norsmalllen']
    timepoint_id_int = sio.loadmat(path_timepoint_id_int)
    timepoint_id_int = timepoint_id_int['timepoint_id_int'].reshape(-1)
    label = sio.loadmat(path_label)
    label = label['label'].reshape(-1)
    num_data = len(fmri_signal)
    # print(label.shape)
    label_values, label_counts = np.unique(label, return_counts=True)
    print("num of baseline labels: ", label_values, label_counts)
    dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
    print("num of dx labels: ", dx_label_values, dx_label_counts)
    fmri_subid_values, fmri_subid_counts = np.unique(fmri_subid, return_counts=True)
    print("num of subject: ", len(fmri_subid_counts))
    print("num of subject with bl signal: ", np.sum(index_bl))
    print("num of subject with timepoint >=0 : ", np.sum(timepoint_id_int >= 0))
    # get_label_conversion(dx_label, fmri_subid, timepoint_id_int, index_bl)
    isbaseline, conversion_label = get_label_conversion_wobl(dx_label, fmri_subid, timepoint_id_int)

    normalized_recons_sig = np.load(os.path.join(root, "normalized_recons_sig_bySDE.npy"))
    print("reconstructed signal shape: ", normalized_recons_sig.shape)
    pick_normalized_recons_sig, pick_fmri_subid, pick__label = sample_recons_sig_bylabels(normalized_recons_sig, dx_label, fmri_subid, isbaseline, conversion_label, disease_id=disease_id,
                               isUseDXLabelOnly=isUseDXLabelOnly, isUseDXLabelwithBaselineOnly=isUseDXLabelwithBaselineOnly, isTestConversion=isTestConversion)
    return pick_normalized_recons_sig, pick_fmri_subid, pick__label

def sample_recons_sig_bylabels(normalized_data, dx_label, fmri_subid, isbaseline, conversion_label, disease_id=0, isUseDXLabelOnly=True, isUseDXLabelwithBaselineOnly = False,
                               isTestConversion = False):
    print("select disease id: %d; isUseDXLabelOnly: " % (disease_id), isUseDXLabelOnly, "; isUseDXLabelwithBaselineOnly:", isUseDXLabelwithBaselineOnly, "; isTestConversion:", isTestConversion)
    print("check correctness: len of isbaseline and conversion_label", isbaseline.shape, conversion_label.shape)
    if isUseDXLabelOnly:
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: %d/%d, %d/%d"%(normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
        print("check correctness of shape: normalized_data, fmri_subid, dx_label", normalized_data.shape, fmri_subid.shape, dx_label.shape)
        return normalized_data, fmri_subid, dx_label
    elif isUseDXLabelwithBaselineOnly:
        normalized_data, dx_label, fmri_subid = normalized_data[isbaseline], dx_label[isbaseline], fmri_subid[isbaseline]
        dx_label_values, dx_label_counts = np.unique(dx_label, return_counts=True)
        print("UseDXLabelwithBaselineOnly: num of dx labels: ", dx_label_values, dx_label_counts)
        select_indices = np.arange(len(normalized_data))
        if disease_id == 0:
            select_indices = np.where((dx_label == 0) | (dx_label == 2))[0]
        elif disease_id == 1:
            select_indices = np.where((dx_label == 0) | (dx_label == 1))[0]
        elif disease_id == 2:
            select_indices = np.where((dx_label == 1) | (dx_label == 2))[0]
        else:
            select_indices = np.where((dx_label >= 0))[0]

        normalized_data = normalized_data[select_indices]
        fmri_subid = fmri_subid[select_indices]
        dx_label = dx_label[select_indices]

        if disease_id == 2:
            dx_label[dx_label != 2] = 0
            dx_label[dx_label == 2] = 1
        else:
            dx_label[dx_label > 0] = 1

        normal_sum = np.sum(dx_label == 0)
        abnormal_sum = np.sum(dx_label > 0)
        print("Num of normal and abnormal samples: %d/%d, %d/%d" % (
        normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
        print("check correctness of shape: normalized_data, fmri_subid, dx_label", normalized_data.shape,
              fmri_subid.shape, dx_label.shape)
        return normalized_data, fmri_subid, dx_label
    elif isTestConversion:
        pos_hasmultiple_sig = conversion_label>=0
        normalized_data, dx_label, fmri_subid, conversion_label = normalized_data[pos_hasmultiple_sig], dx_label[pos_hasmultiple_sig], fmri_subid[
            pos_hasmultiple_sig], conversion_label[pos_hasmultiple_sig]

        conversion_label[conversion_label>=1]=1

        normal_sum = np.sum(conversion_label == 0)
        abnormal_sum = np.sum(conversion_label > 0)
        print("Num of non-conversion and conversion samples: %d/%d, %d/%d" % (
            normal_sum, len(normalized_data), abnormal_sum, len(normalized_data)))
        print("check correctness of shape: normalized_data, fmri_subid, conversion_label", normalized_data.shape,
              fmri_subid.shape, conversion_label.shape)
        return normalized_data, fmri_subid, conversion_label

class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval
    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)
    @property
    def val(self):
        return self._val

def normalize_sigdata(num_data, fmri_signal, maxlen=197, isSelfNormalize=False, isPadding=True,
                      UseReconstructSignal=False):
    len_raw_signal = []
    normalized_data = []
    if isSelfNormalize and not UseReconstructSignal:
        normalized_data = []
        scaler = StandardScaler()
        for i in range(num_data):
            tmp_sig = fmri_signal[i][0]
            scaler.fit(tmp_sig)
            signal_sub = scaler.transform(tmp_sig)
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant')
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
    else:
        all_data = []
        for i in range(num_data):
            if UseReconstructSignal:
                tmp = fmri_signal[i]
                tmp = np.transpose(tmp, (1,0))
                all_data.append(tmp)
            else:
                all_data.append(fmri_signal[i][0])
        all_data = np.concatenate(all_data)
        scaler = StandardScaler()
        scaler.fit(all_data)
        # print(all_data.shape)

        normalized_data = []
        for i in range(num_data):
            if UseReconstructSignal:
                tmp = fmri_signal[i]
                signal_sub = np.transpose(tmp, (1, 0))
            else:
                signal_sub = fmri_signal[i][0]
            signal_sub = scaler.transform(signal_sub)
            len_raw_signal.append(len(signal_sub))
            if isPadding and len(signal_sub) < maxlen:
                mean_values = np.mean(signal_sub, 0)
                mean_values = np.tile(mean_values, (maxlen - len(signal_sub), 1))
                # signal_sub = np.pad(np.asarray(signal_sub), [(0, maxlen - len(signal_sub)), (0, 0)],
                #                     mode='constant',constant_values = (0,0))
                signal_sub = np.concatenate((signal_sub, mean_values), 0)
            signal_sub = signal_sub[:maxlen]
            signal_sub = np.expand_dims(signal_sub, 0)
            normalized_data.append(signal_sub)
    normalized_data = np.concatenate(normalized_data)
    normalized_data = np.transpose(normalized_data, (0, 2, 1))
    len_raw_signal = np.asarray(len_raw_signal)
    print("signal min length:%d" % (np.min(len_raw_signal)))
    print("num of small len: %d/%d ;len %d: %d/%d" % (
        num_data - np.sum(len_raw_signal >= maxlen), num_data, maxlen, num_data - np.sum(len_raw_signal < maxlen),
        num_data))
    return normalized_data, len_raw_signal

def chooseReconstrctDataBySDE(raw_data = None, control_std = 3, path_data='./data/sde/reconst_sig', train_len=140, batch_size=2048, max_iters = 900,
                              num_sub_small_sig = 681, rois=100, maxlen=197, index_small_len=None, select_normalized_data=None, select_fmri_subid=None):
    info_path = os.path.join(path_data, 'all_ADNI')
    recons_path = os.path.join(path_data, 'saved_file/ckpts')
    device = torch.device('cpu')
    batch_fmri_subid = []
    batch_normalized_sig_list = []
    batch_signal_mean_std_list = []
    reconst_sig_list = []

    for index_start_point in range(math.ceil(num_sub_small_sig*rois/batch_size)):
        start_point = index_start_point * batch_size
        recons_file_name = os.path.join(recons_path, 'start_%d_global_step_%d_Reconstruct_True.ckpt'%(index_start_point, max_iters))
        ckpt = torch.load(recons_file_name, map_location=device)
        copy_zs_from_sde = ckpt['copy_zs'].numpy()
        copy_zs_from_sde = np.transpose(copy_zs_from_sde, (1,0))
        reconst_sig_list.append(copy_zs_from_sde)
        tmp_subid = np.load(os.path.join(info_path, 'batch_fmri_subid_start_sig%d.npy'%(start_point)))
        tmp_normalized_sig = np.load(os.path.join(info_path, 'batch_normalized_sig_list_start_sig%d.npy'%(start_point)))
        tmp_signal_mean_std = np.load(os.path.join(info_path, 'batch_signal_mean_std_list_start_sig%d.npy'%(start_point)))
        batch_fmri_subid.append(tmp_subid)
        batch_normalized_sig_list.append(tmp_normalized_sig)
        batch_signal_mean_std_list.append(tmp_signal_mean_std)
    batch_fmri_subid = np.concatenate(batch_fmri_subid)
    batch_normalized_sig_list = np.concatenate(batch_normalized_sig_list)
    batch_signal_mean_std_list = np.concatenate(batch_signal_mean_std_list)
    reconst_sig_list = np.concatenate(reconst_sig_list)

    batch_fmri_subid = np.reshape(batch_fmri_subid, (num_sub_small_sig, rois))
    batch_fmri_subid = batch_fmri_subid[:, 0]
    batch_normalized_sig_list = np.reshape(batch_normalized_sig_list, (num_sub_small_sig, rois, maxlen))
    batch_signal_mean_std_list = np.reshape(batch_signal_mean_std_list, (num_sub_small_sig, rois, 2, 100))
    reconst_sig_list = np.reshape(reconst_sig_list, (num_sub_small_sig, rois, maxlen))

    ori_sig_list = getback_orisig_bymeanstd(batch_normalized_sig_list, batch_signal_mean_std_list)

    ####### combime ori sig and the reconstructed one by sde ##########
    recons_sig_list = batch_normalized_sig_list.copy()
    recons_sig_list[:, :, train_len:] = reconst_sig_list[:, :, train_len:] * control_std
    recons_sig_list = getback_orisig_bymeanstd(recons_sig_list, batch_signal_mean_std_list)

    # visualize_ori_recons_sig(ori_sig_list, recons_sig_list, tmp_roi=0, tmp_subid=0)

    all_recons_sig = raw_data.copy()
    all_recons_sig[index_small_len] = recons_sig_list

    ####### check correctness ##########
    error_fmri_subid = np.sum(select_fmri_subid - batch_fmri_subid)
    error_fmri_signal = np.sum(select_normalized_data[:, :, :train_len] - recons_sig_list[:, :, :train_len])
    error_fmri_all_signal = np.sum(all_recons_sig[:, :, :train_len] - raw_data[:, :, :train_len])
    print("check correctness error code of sde :%d, %d, %d (0 means correct)"%(error_fmri_subid, error_fmri_signal, error_fmri_all_signal))
    print("new signal min_max:",np.min(all_recons_sig),np.max(all_recons_sig), "num of small value:", np.sum(all_recons_sig<3.0)," new shape:", all_recons_sig.shape)

    # visualize_ori_recons_sig(all_recons_sig, all_recons_sig, tmp_roi=0, tmp_subid=2)
    return all_recons_sig

def getback_orisig_bymeanstd(batch_normalized_sig_list, batch_signal_mean_std_list):
    batch_normalized_sig_list = np.transpose(batch_normalized_sig_list, (0, 2, 1))
    batch_signal_mean_std_list = batch_signal_mean_std_list[:, 0, :, :]
    ori_sig_list = []
    for i in range(len(batch_normalized_sig_list)):
        each_signal = batch_normalized_sig_list[i]
        mean_ = batch_signal_mean_std_list[i, 0]
        std_ = batch_signal_mean_std_list[i, 1]
        ori_sig = each_signal * std_ + mean_
        ori_sig = np.expand_dims(ori_sig, 0)
        ori_sig_list.append(ori_sig)
    ori_sig_list = np.concatenate(ori_sig_list)
    ori_sig_list = np.transpose(ori_sig_list, (0, 2, 1))
    return ori_sig_list

def visualize_ori_recons_sig(ori_sig_list, recons_sig_list, tmp_roi=10, tmp_subid=600):
    # tmp_roi = 10
    # tmp_subid = 600
    choose_sig = ori_sig_list[tmp_subid, tmp_roi]
    choose_recons_sig = recons_sig_list[tmp_subid, tmp_roi]
    plt.subplot(frameon=False)
    plt.plot(range(len(choose_sig)), choose_sig, marker='x', color='grey', label='raw data')
    plt.plot(range(len(choose_recons_sig)), choose_recons_sig, color='#fc4e2a', linewidth=1.0)
    plt.xlabel('$t$')
    plt.ylabel('$Y_t$')
    plt.tight_layout()
    plt.show()

class SignalDataset(Dataset):
    def __init__(self, X, y, mask=None, adj=None):
        self.data = torch.from_numpy(X).float()
        self.target = torch.from_numpy(y).long()
        self.mask = mask
        if mask is not None:
            self.mask = torch.from_numpy(mask).long()
        self.adj = adj
        if adj is not None:
            self.adj = torch.from_numpy(adj).float()
        self.length = self.target.shape[0]

    def __getitem__(self, index):
        if self.mask is None and self.adj is None:
            return self.data[index], self.target[index]
        elif self.mask is None and self.adj is not None:
            return self.data[index], self.target[index], self.adj[index]
        elif self.adj is not None:
            return self.data[index], self.target[index], self.mask[index], self.adj[index]
        else:
            return self.data[index], self.target[index], self.mask[index]

    def __len__(self):
        return self.length

    def get_labels(self):
        return self.target

if __name__ == '__main__':
    root = './data'
    # load_fmri(root, disease_id=1, isSelfNormalize=True, isDrawSection=False, isUseDXLabel=True)
    #load_fmri_AllADNI(root, disease_id=0, isSelfNormalize=True, isDrawSection=False, isUseDXLabel=True)
    # load_fmri_AllADNI(root, isNormalize=False, disease_id=0, isSelfNormalize=True, isDrawSection=False, isUseDXLabel=True,
    #                   isSelect4Reconstruct = True, maxlen=197, UseReconstructSignal=True)

    load_recons_fmri_AllADNI(root, disease_id=0, isUseDXLabelOnly=False, isUseDXLabelwithBaselineOnly = False,
                            isTestConversion = True)

