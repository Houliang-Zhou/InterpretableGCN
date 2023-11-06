import os
import logging
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import random
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

def show_signal(data, recons_signal, fontsize=24, label='data', subid=0, roi_id=0, args=None, epoch=0, isSaved=True):
    if isSaved:
        score_result = data.detach().cpu().numpy()
        result_path = os.path.join(args.res_dir, 'rnn_raw_data_epoch%d.npy' % (epoch))
        with open(result_path, 'wb') as f:
            np.save(f, score_result)
        score_result = recons_signal.detach().cpu().numpy()
        result_path = os.path.join(args.res_dir, 'rnn_recons_signal_epoch%d.npy' % (epoch))
        with open(result_path, 'wb') as f:
            np.save(f, score_result)
    data = data[subid, roi_id]
    recons_signal = recons_signal[subid, roi_id]
    #print("data std: raw %f, recons %f"%(data.view(-1).std(), recons_signal.view(-1).std()))
    fig=plt.figure() #, figsize=(24, 12)
    plt.plot(data.view(-1).detach().cpu().numpy(), label='raw data')
    plt.plot(recons_signal.view(-1).detach().cpu().numpy(), label='reconstructed')
    plt.ylabel("Value", fontsize=fontsize // 2)
    plt.xlabel("Timesteps", fontsize=fontsize // 2)
    # plt.legend(labels=['Consumed', 'Exist'], loc='upper left')
    # plt.xticks(rotation=45)
    # plt.savefig('./pic/%s_number.png' % (title_name), dpi=600)
    result_file_name = "rnn_result_imgs_sub0_time0"
    result_path = os.path.join(args.res_dir, '%s_epoch%d.png' % (result_file_name, epoch))
    if isSaved:
        plt.savefig('%s' % (result_path), dpi=600)
    else:
        plt.show()
    plt.close(fig)

def show_signal_bybatch(data, recons_signal, fontsize=24, label='data', subid=0, args=None, epoch=0, isSaved=True):
    if isSaved:
        score_result = data.detach().cpu().numpy()
        result_path = os.path.join(args.res_dir, 'raw_data_epoch%d.npy' % (epoch))
        with open(result_path, 'wb') as f:
            np.save(f, score_result)
        score_result = recons_signal.detach().cpu().numpy()
        result_path = os.path.join(args.res_dir, 'recons_signal_epoch%d.npy' % (epoch))
        with open(result_path, 'wb') as f:
            np.save(f, score_result)
    data = data[subid]
    recons_signal = recons_signal[subid]
    #print("data std: raw %f, recons %f"%(data.view(-1).std(), recons_signal.view(-1).std()))
    fig=plt.figure() #, figsize=(24, 12)
    plt.plot(data.view(-1).detach().cpu().numpy(), label='raw data')
    plt.plot(recons_signal.view(-1).detach().cpu().numpy(), label='reconstructed')
    plt.ylabel("Value", fontsize=fontsize // 2)
    plt.xlabel("Timesteps", fontsize=fontsize // 2)
    # plt.legend(labels=['Consumed', 'Exist'], loc='upper left')
    # plt.xticks(rotation=45)
    result_file_name = "result_imgs_sub0_time0"
    result_path = os.path.join(args.res_dir, '%s_epoch%d.png' % (result_file_name, epoch))
    if isSaved:
        plt.savefig('%s' % (result_path), dpi=600)
    else:
        plt.show()
    plt.close(fig)