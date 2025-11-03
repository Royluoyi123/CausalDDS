
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Union
from sklearn.metrics import roc_auc_score,auc, roc_curve, average_precision_score,cohen_kappa_score, f1_score, accuracy_score,precision_recall_curve,balanced_accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

def get_stats(array):
    
    mean = np.mean(np.asarray(array))
    std = np.std(np.asarray(array))

    return mean, std


def write_summary(args, config_str, stats):
    
    f = open("DDS/results/{}/10/{}/lam1_{}_lam2_{}_lr_{}_wd_{}_drop_{}_f1.txt".format(args.dataset,args.setting,args.lam1,args.lam2,args.lr,args.weight_decay, args.dropout), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("AUC : {:.4f} || AUPR : {:.4f} || Acc : {:.4f} || Kappa : {:.4f} || Bacc : {:.4f}|| F1 : {:.4f}".format(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary_reg(args, config_str, stats):
    
    f = open("DDS/try/{}/regression/{}/lam1_{}_lam2_{}_lr_{}_wd_{}_drop_{}_f1.txt".format(args.dataset,args.setting,args.lam1,args.lam2,args.lr,args.weight_decay, args.dropout), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("MSE : {:.4f} || RMSE : {:.4f} || R2 : {:.4f} || PEAR : {:.4f} || MAE : {:.4f}|| SPEAR : {:.4f}".format(stats[0], stats[1], stats[2], stats[3], stats[4], stats[5]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_summary_total(args, config_str, stats):
    
    f = open("DDS/results/{}/10/{}/lam1_{}_lam2_{}_lr_{}_wd_{}_drop_{}_total.txt".format(args.dataset,args.setting,args.lam1,args.lam2,args.lr,args.weight_decay, args.dropout), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("AUC : {:.4f}({:.4f}) || AUPR : {:.4f}({:.4f}) || ACC : {:.4f}({:.4f}) || KAPPA : {:.4f}({:.4f})|| BACC : {:.4f}({:.4f}) || F1 : {:.4f}({:.4f})".format(stats[0], stats[1], stats[2], stats[3],
                                                                                                                                    stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()

def write_summary_total_reg(args, config_str, stats):
    
    f = open("DDS/results/{}/regression/{}/lam1_{}_lam2_{}_lr_{}_wd_{}_drop_{}_total_f1.txt".format(args.dataset,args.setting,args.lam1,args.lam2,args.lr,args.weight_decay, args.dropout), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write("MSE : {:.4f}({:.4f}) || RMSE : {:.4f}({:.4f}) || R2 : {:.4f}({:.4f}) || PEAR : {:.4f}({:.4f})|| MAE : {:.4f}({:.4f}) || SPEAR : {:.4f}({:.4f})".format(stats[0], stats[1], stats[2], stats[3],
                                                                                                                                    stats[4], stats[5], stats[6], stats[7], stats[8], stats[9], stats[10], stats[11]))
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def write_experiment(args, config_str, best_config):
    
    f = open("DDS/results/{}/{}.txt".format(args.dataset, args.setting), "a")
    f.write("--------------------------------------------------------------------------------- \n")
    f.write(config_str)
    f.write("\n")
    f.write(best_config)
    f.write("\n")
    f.write("--------------------------------------------------------------------------------- \n")
    f.close()


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


def get_len_matrix(len_list):
    len_list = np.array(len_list)
    max_nodes = np.sum(len_list)
    curr_sum = 0
    len_matrix = []
    for l in len_list:
        curr = np.zeros(max_nodes)
        curr[curr_sum:curr_sum + l] = 1
        len_matrix.append(curr)
        curr_sum += l
    return np.array(len_matrix)


def create_batch_mask(samples):
    batch0 = samples[0].batch.reshape(1, -1)
    index0 = torch.cat([batch0, torch.tensor(range(batch0.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (batch0.max() + 1, batch0.shape[1]))

    batch1 = samples[1].batch.reshape(1, -1)
    index1 = torch.cat([batch1, torch.tensor(range(batch1.shape[1])).reshape(1, -1)])
    mask1 = torch.sparse_coo_tensor(index1, torch.ones(index1.shape[1]), size = (batch1.max() + 1, batch1.shape[1])) #创建矩阵 mask0 药物1 mask1药物2

    return mask0, mask1


def create_interv_mask(rand_batch):

    rand_batch = rand_batch.reshape(1, -1)
    index0 = torch.cat([rand_batch, torch.tensor(range(rand_batch.shape[1])).reshape(1, -1)])
    mask0 = torch.sparse_coo_tensor(index0, torch.ones(index0.shape[1]), size = (rand_batch.max() + 1, rand_batch.shape[1]))

    return mask0


class KLD(nn.Module):
    def forward(self, inputs, targets):

        inputs = F.log_softmax(inputs, dim=0)
        targets = F.softmax(targets, dim=0)
        
        return F.kl_div(inputs, targets, reduction='batchmean')


def get_roc_score(preds, labels):

    auc_score = AUC(labels, preds)
    aupr_score = AUPR(labels, preds)
    acc_score = ACC(labels, preds)
    kappa_score = KAPPA(labels, preds)
    bacc_score = BACC(labels, preds)
    f1_score = F1(labels, preds)

    return auc_score, aupr_score, acc_score, kappa_score, bacc_score, f1_score


def get_mse_score(preds, labels):

    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    pearson_corr, pearson_p = pearsonr(labels, preds)
    spearman_corr, spearman_p = spearmanr(labels, preds)

    return mse, rmse, pearson_corr, r2 ,mae, spearman_corr

def AUC(ground_truth, prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.
    return res

def AUPR(ground_truth,prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    precision,recall,thresholds=precision_recall_curve(ground_truth, prediction)
    res=auc(recall,precision)
    #res = average_precision_score(y_true=ground_truth, y_score=prediction)
    return(res)

def ACC(ground_truth,prediction):
    prediction=[0 if np.isnan(x) else x for x in prediction]
    #fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    
    #youden_thresh = round(youden_thresh, 3)
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]
    return accuracy_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def KAPPA(ground_truth,prediction):
    #fpr, tpr, threshold = roc_curve(ground_truth, prediction)
    # 利用Youden's index计算阈值
    #spc = 1 - fpr
    #j_scores = tpr - fpr
    #best_youden, youden_thresh, youden_sen, youden_spc = sorted(zip(j_scores, threshold, tpr, spc))[-1]
    
    #youden_thresh = round(youden_thresh, 3)

    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]
    
    return cohen_kappa_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def BACC(ground_truth,prediction):
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)

    # 计算每个 threshold 对应的 F1-score
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # 避免除以0

    # 找到 F1-score 最大的索引
    best_idx = f1_scores.argmax()

    # 获取最佳阈值
    best_threshold = thresholds[best_idx]

    return balanced_accuracy_score(ground_truth,[int(i>=best_threshold) for i in prediction ])

def F1(ground_truth, prediction):
    precision, recall, thresholds = precision_recall_curve(ground_truth, prediction)
    
    # 计算每个阈值下的F1-score（加1e-10防止除零）
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # 返回最大的F1值
    return f1_scores.max()

def eval_threshold(labels_all, preds_all):

    # fpr, tpr, thresholds = roc_curve(labels_all, preds_all)
    # optimal_idx = np.argmax(tpr - fpr)
    # optimal_threshold = thresholds[optimal_idx]
    optimal_threshold = 0.5
    preds_all_ = []
    
    for p in preds_all:
        if p >=optimal_threshold:
            preds_all_.append(1)
        else:
            preds_all_.append(0)

    return preds_all, preds_all_