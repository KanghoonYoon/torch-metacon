import os
import re
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='citeseer', \
    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')

parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
parser.add_argument('--split', nargs='+', type=float, default=[0.1, 0.1, 0.1])
parser.add_argument('--attack_method', type=str, default='metacon', choices=['metacon', 'metacon+', 'Meta-Self', 'GraD'])


## MetaAttack Parameters
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--inner_train_iters', type=int, default=500)
parser.add_argument('--momentum', type=float, default=0.9)

## Metacon Parameters
parser.add_argument('--coef1', type=float, default=1.0)
parser.add_argument('--coef2', type=float, default=0.05)
parser.add_argument('--droprate1', type=float, default=0.0)
parser.add_argument('--droprate2', type=float, default=0.1)

parser.add_argument('--savepath', type=str, default="result")



args = parser.parse_args()


path = f'{args.savepath}/summary_result/{args.dataset}_ptb({args.ptb_rate:.2f})_split({args.split[0]})'

# models=['meta_victim(gcn)', 'meta_victim(graphsage)', 'meta_victim(gat)']

models=[f'/{args.attack_method}_victim(gcn)', f'/{args.attack_method}_victim(graphsage)', f'/{args.attack_method}_victim(gat)']

for model in models:

    filename = path + model + '.txt'
    with open(filename, 'r') as f:
        file = f.readlines()

    if file[-1].startswith('Mean'):
        exit()

    accs = [d for d in file if d.startswith('Clean')]
    accs = np.array([re.findall(r"[-+]?(?:\d*\.*\d+)", accs[i]) for i in range(len(accs))]).astype('float')

    cnts = [d for d in file if d.startswith('Num Attack')]
    cnts = np.array([re.findall(r"[-+]?(?:\d*\.*\d+)", cnts[i]) for i in range(len(cnts))]).astype('float')

    acc_mean = np.mean(accs, 0) 
    acc_std = np.std(accs, 0)

    cnt_mean = np.mean(cnts, 0) 
    cnt_std = np.std(cnts, 0)

    if acc_mean.shape[0] == 4:
        result = f"Mean Clean ACC: {100*acc_mean[0]:.2f}" + u"\u00B1" + f"{100*acc_std[0]:.2f}  Mean Evasive ACC: {100*acc_mean[1]:.2f}" + u"\u00B1" + f"{100*acc_std[1]:.2f}  Mean Poi-Tr ACC: {100*acc_mean[2]:.2f}" + u"\u00B1" + f"{100*acc_std[2]:.2f}" \
            + f"  Poi-Val ACC: {100*acc_mean[3]:.2f}" + u"\u00B1" + f"{100*acc_std[3]:.2f}\n" \
            + f"Mean Num Attack LL: {int(cnt_mean[0]):d}" + u"\u00B1" + f"{cnt_std[0]:.2f}  LU: {int(cnt_mean[1]):d}" + u"\u00B1" + f"{cnt_std[1]:.2f}  UU: {int(cnt_mean[2]):d}" + u"\u00B1" + f"{cnt_std[2]:.2f}  " \
            + f"Mean Prop. Attack LL: {cnt_mean[3]:.3f}" + u"\u00B1" + f"{cnt_std[3]:.2f}  LU: {cnt_mean[4]:.2f}" + u"\u00B1" + f"{cnt_std[4]:.2f}  UU: {cnt_mean[5]:.2f}" + u"\u00B1" + f"{cnt_std[5]:.2f}"
    else:
        result = f"Mean Clean ACC: {100*acc_mean[0]:.2f}" + u"\u00B1" + f"{100*acc_std[0]:.2f}  Mean Evasive ACC: {100*acc_mean[1]:.2f}" + u"\u00B1" + f"{100*acc_std[1]:.2f}  Mean Poi-Tr ACC: {100*acc_mean[2]:.2f}" + u"\u00B1" + f"{100*acc_std[2]:.2f}" \
            + f"  Poi-Val ACC: nan\n" \
            + f"Mean Num Attack LL: {int(cnt_mean[0]):d}" + u"\u00B1" + f"{cnt_std[0]:.2f}  LU: {int(cnt_mean[1]):d}" + u"\u00B1" + f"{cnt_std[1]:.2f}  UU: {int(cnt_mean[2]):d}" + u"\u00B1" + f"{cnt_std[2]:.2f}  " \
            + f"Mean Prop. Attack LL: {cnt_mean[3]:.2f}" + u"\u00B1" + f"{cnt_std[3]:.2f}  LU: {cnt_mean[4]:.2f}" + u"\u00B1" + f"{cnt_std[4]:.2f}  UU: {cnt_mean[5]:.2f}" + u"\u00B1" + f"{cnt_std[5]:.2f}"
            

    with open(filename, 'a') as f:
        f.write('============================================================\n' \
            + result \
                + '\n============================================================\n')
