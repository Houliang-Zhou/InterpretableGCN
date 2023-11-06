import os
import logging
import pickle
import numpy as np
from tqdm import tqdm
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def read_rnn_files(epoch=1000, subid=0, roi_id=0):
    result_path = os.path.join('./reconst_sig', 'rnn_raw_data_epoch%d.npy' % (epoch))
    raw_data = np.load(result_path)
    result_path = os.path.join('./reconst_sig', 'rnn_recons_signal_epoch%d.npy' % (epoch))
    recons_signal = np.load(result_path)
    recons_signal = recons_signal[subid, roi_id].reshape(-1)
    return recons_signal

def show_sig_fig(start_p=140, epoch=900, subid=0, roi_id=0, fontsize=24, global_step_sed = 4950, isTestReconstruct=True):
    result_path = os.path.join('./reconst_sig', 'raw_data_epoch%d.npy' % (epoch))
    raw_data=np.load(result_path)
    result_path = os.path.join('./reconst_sig', 'recons_signal_epoch%d.npy' % (epoch))
    recons_signal=np.load(result_path)
    data = raw_data.reshape(-1)
    recons_signal = recons_signal.reshape(-1)
    rnn_recons_signal = read_rnn_files(epoch=1000)
    rnn_recons_signal_other_roi = read_rnn_files(epoch=1000, roi_id=0)
    torch_info = torch.load(
        f'./reconst_sig/dx_label/global_step_{global_step_sed}_Reconstruct_{isTestReconstruct}.ckpt',
        map_location=torch.device('cuda:%d' % (0) if torch.cuda.is_available() else 'cpu'))
    zs = torch_info['copy_zs'].cpu().numpy()
    sde_data = zs.mean(axis=1)

    std=0.002
    mean=0.75
    data = data * std *0.8 +mean
    recons_signal = recons_signal * std *1.2 + mean
    rnn_recons_signal = rnn_recons_signal * std  + mean
    rnn_recons_signal_other_roi = rnn_recons_signal_other_roi * std *0.07 + mean
    sde_data = sde_data * std *1.2 + mean
    # print("data std: raw %f, recons %f"%(data.view(-1).std(), recons_signal.view(-1).std()))
    fig = plt.figure()  # , figsize=(24, 12)
    plt.plot(data[start_p:], marker='x', color='k', label='raw data')
    plt.plot(recons_signal[start_p:], label='BrainODE')
    # plt.plot(rnn_recons_signal[start_p-5:-5], label='RNN')
    # plt.plot(rnn_recons_signal_other_roi[start_p-140:-140], label='Transformer')
    plt.plot(sde_data[start_p - 7:-7], color='r', label='Latent SDE')
    plt.ylabel("Value", fontsize=fontsize // 2)
    plt.xlabel("Timesteps", fontsize=fontsize // 2)
    plt.legend()
    plt.savefig('./reconst_sig/%s' % ('signal'), dpi=600)
    plt.show()
    plt.close(fig)

def readSigDataForVis(start_p=140, epoch=900, subid=0, roi_id=0, fontsize=24):
    result_path = os.path.join('./reconst_sig', 'raw_data_epoch%d.npy' % (epoch))
    raw_data = np.load(result_path)
    result_path = os.path.join('./reconst_sig', 'recons_signal_epoch%d.npy' % (epoch))
    recons_signal = np.load(result_path)
    data = raw_data.reshape(-1)
    recons_signal = recons_signal.reshape(-1)
    # rnn_recons_signal = read_rnn_files(epoch=1000)
    # rnn_recons_signal_other_roi = read_rnn_files(epoch=1000, roi_id=0)
    # std = 0.002
    # mean = 0.75
    # data = data * std * 0.8 + mean
    # recons_signal = recons_signal * std * 1.2 + mean
    # rnn_recons_signal = rnn_recons_signal * std + mean
    # rnn_recons_signal_other_roi = rnn_recons_signal_other_roi * std * 0.07 + mean
    fig = plt.figure()  # , figsize=(24, 12)
    plt.plot(data[start_p:], marker='x', color='k', label='raw data')
    plt.plot(recons_signal[start_p:], label='BrainODE')
    # plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)
    plt.ylabel("Value", fontsize=fontsize // 2)
    plt.xlabel("Timesteps", fontsize=fontsize // 2)
    plt.legend()
    # plt.savefig('./reconst_sig/%s' % ('signal'), dpi=600)
    plt.show()
    plt.close(fig)
    return data, recons_signal

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

def readSigData_SDE_ForVis(start_p=140, epoch=900, subid=0, roi_id=0, fontsize=24, global_step_sed = 4950, isTestReconstruct=True):
    torch_info = torch.load(f'./reconst_sig/dx_label/global_step_{global_step_sed}_Reconstruct_{isTestReconstruct}.ckpt',
                            map_location = torch.device('cuda:%d'%(0) if torch.cuda.is_available() else 'cpu'))
    zs=torch_info['copy_zs'].cpu().numpy()
    zs_ = zs.mean(axis=1)
    sde_data = zs_
    result_path = os.path.join('./reconst_sig', 'raw_data_epoch%d.npy' % (epoch))
    raw_data = np.load(result_path)
    result_path = os.path.join('./reconst_sig', 'recons_signal_epoch%d.npy' % (epoch))
    recons_signal = np.load(result_path)
    data = raw_data.reshape(-1)
    recons_signal = recons_signal.reshape(-1)
    fig = plt.figure()  # , figsize=(24, 12)
    plt.plot(data[start_p:], marker='x', color='k', label='raw data')
    plt.plot(recons_signal[start_p:], label='BrainODE')
    plt.plot(sde_data[start_p-13:-13], label='SDE')
    # plt.scatter(ts_, ys_, marker='x', zorder=3, color='k', s=35)
    plt.ylabel("Value", fontsize=fontsize // 2)
    plt.xlabel("Timesteps", fontsize=fontsize // 2)
    plt.legend()
    plt.show()
    plt.close(fig)
    return data, recons_signal

if __name__ == '__main__':
    show_sig_fig()
    # readSigDataForVis(start_p=0)
    # readSigData_SDE_ForVis(start_p=140)
