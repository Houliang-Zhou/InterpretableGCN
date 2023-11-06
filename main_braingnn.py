import os.path as osp
import os, sys
import time
from shutil import copy, rmtree
from itertools import product
import pdb
import argparse
import random
import torch
import numpy as np
from kernel.train_eval_braingnn import cross_validation_with_val_set
from kernel.train_eval_braingnn import cross_validation_without_val_set
from utils import *
from data import *

# used to traceback which code cause warnings, can delete
import traceback
import warnings
import sys
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    log = file if hasattr(file,'write') else sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))

warnings.showwarning = warn_with_traceback


# General settings.
parser = argparse.ArgumentParser(description='BrainGNN for ADNI graphs')
parser.add_argument('--data', type=str, default='SGCN')
parser.add_argument('--clean', action='store_true', default=False,
                    help='use a cleaned version of dataset by removing isomorphism')
parser.add_argument('--no_val', action='store_true', default=False,
                    help='if True, do not use validation set, but directly report best\
                    test performance.')
parser.add_argument('--disease_id', type=int, default=1,
                    help='disease_id for classification: HCvs.AD: 0; HCvs.MCI: 1; MCIvs.AD:2')
parser.add_argument('--isUseDXLabelOnly', action='store_true', default=True,
                    help='isUseDXLabelOnly')
parser.add_argument('--isUseDXLabelwithBaselineOnly', action='store_true', default=False,
                    help='isUseDXLabelwithBaselineOnly')
parser.add_argument('--isTestConversion', action='store_true', default=False,
                    help='isTestConversion')


parser.add_argument('--training_len', type=int, default=197)
parser.add_argument('--max_len', type=int, default=197)

parser.add_argument('--isUseSampler', action='store_true', default=True,
                    help='Use Sampler for imbalanced data')
parser.add_argument('--isCosineAnnealingLR', action='store_true', default=False,
                    help='use CosineAnnealingLR')

parser.add_argument('--l2', type=float, default=5e-3, help='l2 regulazer') #1e-3
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')

#BrainGNN
parser.add_argument('--build_graph_bycorr', action='store_true', default=True,
                    help='build_graph_bycorr')
parser.add_argument('--thredgraph', type=float, default=0.3)
parser.add_argument('--topk_ratio', type=float, default=0.7, help='ratio of topk_nodes in BrainGNN')
parser.add_argument('--lamda_x_l1', type=float, default=0.1)
parser.add_argument('--lamda_e_l1', type=float, default=0.1)
parser.add_argument('--lamda_x_ent', type=float, default=0.1)
parser.add_argument('--lamda_e_ent', type=float, default=0.1)
parser.add_argument('--lamda_mi', type=float, default=1.0)
parser.add_argument('--lamda_ce', type=float, default=1.0)
parser.add_argument('--lamda_prob', type=float, default=1.0)

parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 entropy regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 entropy regularization')
parser.add_argument('--lamb5', type=float, default=0.1, help='s1 consistence regularization')
parser.add_argument('--ratio', type=float, default=0.5, help='pooling ratio')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')

# GNN settings.
parser.add_argument('--model', type=str, default='Network',
                    help='SGCN, GCN, GraphSAGE, GIN, GAT')
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hiddens', type=int, default=10)

# Training settings.
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1E-2) #1e-3
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=50)
parser.add_argument('--fold', type=int, default=5)

# Other settings.
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--keep_files', action='store_true', default=True,
                    help='keep_files')
parser.add_argument('--search', action='store_true', default=False,
                    help='search hyperparameters (layers, hiddens)')
parser.add_argument('--save_appendix', default='',
                    help='what to append to save-names when saving results')
parser.add_argument('--cpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--cuda', type=int, default=0, help='which cuda to use')
args = parser.parse_args()

seed_everything(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

file_dir = os.path.dirname(os.path.realpath('__file__'))
if args.save_appendix == '':
    args.save_appendix = '_' + time.strftime("%Y%m%d%H%M%S")
args.res_dir = os.path.join(file_dir, 'results/ADNI{}'.format(args.save_appendix))
print('Results will be saved in ' + args.res_dir)
if not os.path.exists(args.res_dir):
    os.makedirs(args.res_dir)

if args.keep_files:
    copy('main_braingnn.py', args.res_dir)
    copy('utils.py', args.res_dir)
    copy('kernel/train_eval_braingnn.py', args.res_dir)
    copy('kernel/braingnn.py', args.res_dir)
    copy('data.py', args.res_dir)

# save command line input
cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
    f.write(cmd_input)
print('Command line input: ' + cmd_input + ' is saved.')

if args.data == 'all':
    datasets = [ 'DD', 'MUTAG', 'PROTEINS', 'PTC_MR', 'ENZYMES']
else:
    datasets = [args.data]

if args.search:
    layers = [3, 4]
    hiddens = [10, 16, 32]
else:
    layers = [args.layers]
    hiddens = [args.hiddens]


def logger(info):
    print(info)
    f = open(os.path.join(args.res_dir, 'log.txt'), 'a')
    print(info, file=f)

logger(args)
device = torch.device(
    'cuda:%d'%(args.cuda)  if torch.cuda.is_available() and not args.cpu else 'cpu'
)
args.device = device
print(device)

if args.no_val:
    cross_val_method = cross_validation_without_val_set
else:
    cross_val_method = cross_validation_with_val_set

results = []
for dataset_name in product(datasets):
    best_result = (float('inf'), 0, 0)
    log = '-----\n{}'.format(dataset_name)
    logger(log)
    combinations = product(layers, hiddens)
    best_hyper = (-1, -1)
    for num_layers, hidden in combinations:
        log = "Using {} layers, {} hidden units".format(num_layers, hidden)
        logger(log)
        result_file_name = "result_sgcn_layers{}_hidden{}".format(num_layers, hidden)
        result_path = os.path.join(args.res_dir, '%s.npy'%(result_file_name))
        # max_nodes_per_hop = None
        # data_path = './data/brain_image/knn/%d/'%(args.knn)
        # adni_dataset, len_raw_signal, sig_mask, fmri_subid, target = load_fmri(root='./data', disease_id=args.disease_id)
        adni_dataset, fmri_subid, target = load_recons_fmri_AllADNI(root='./data', disease_id=args.disease_id,
                                                                    isUseDXLabelOnly=args.isUseDXLabelOnly,
                                                                    isUseDXLabelwithBaselineOnly=args.isUseDXLabelwithBaselineOnly,
                                                                    isTestConversion=args.isTestConversion)
        train_dataset = sub_signal(adni_dataset, sub_len = args.max_len)
        # adni_signal_data = SignalDataset(train_dataset, target)
        loss, acc, std  = cross_val_method(
            args,
            train_dataset,
            target,
            folds=args.fold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            gcn_num_layers = num_layers,
            gcn_hidden = hidden,
            weight_decay=0,
            result_path=result_path,
            device=device,
            logger=logger)
        if loss < best_result[0]:
            best_result = (loss, acc, std)
            best_hyper = (num_layers, hidden)

    desc = '{:.3f} ± {:.3f}'.format(
        best_result[1], best_result[2]
    )
    # log = 'Best result - {}, with {} layers and {} hidden units'.format(
    #     desc, best_hyper[0], best_hyper[1]
    # )
    # logger(log)
    results += ['{} : {}'.format(dataset_name, desc)]

log = '-----\n{}'.format('\n'.join(results))
print(cmd_input[:-1])
print(log)
logger(log)
