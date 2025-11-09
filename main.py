import pandas as pd
import torch
import random
import numpy as np
import os
import argument
import time
from utils import get_stats, write_summary, write_summary_total

torch.set_num_threads(2)

def seed_everything(seed=0):
    # To fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def experiment():

    args, unknown = argument.parse_args()
    
    print("Loading dataset...")
    start = time.time()

    # Load dataset

    

    

    

    gene_exp = pd.read_csv('DDS/ddsdata/'+str(args.dataset)+'/cell_line_gene_expression.csv',header=0)
    # 将 DataFrame 的第一列作为字典的 key，剩余列作为 Tensor 作为 value
    gene_exp_dict = {
        row[0]: torch.tensor(row[1:].astype(np.float32)) # 将每行的值转为 Tensor
        for row in gene_exp.to_numpy()
    }
    for args.lam1 in [1]:
        args.lam2 = 1
        for args.lr in [1e-4]:
            for args.weight_decay in [1e-4]:
                for args.dropout in [0.5]:
                    best_aucs, best_auprs, best_accs, best_kappas, best_baccs, best_f1s = [], [], [], [], [], []
                    for repeat in range(1, args.repeat + 1):
                        for fold in range(1, 5):

                            train_set = torch.load("DDS/ddsdata/"+str(args.dataset)+"/processed/classification10/"+args.setting+"/train_"+str(fold)+".pt")
                            test_set = torch.load("DDS/ddsdata/"+str(args.dataset)+"/processed/classification10/"+args.setting+"/test_"+str(fold)+".pt")
            
                            print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
                            stats, config_str, _,  = main(args, train_set, test_set, gene_exp_dict, repeat = repeat, fold = fold)
        
                            # get Stats
                            best_aucs.append(stats[0])
                            best_auprs.append(stats[1])
                            best_accs.append(stats[2])
                            best_kappas.append(stats[3])
                            best_baccs.append(stats[4])
                            best_f1s.append(stats[5])

                            #write_summary(args, config_str, stats)

                            print(fold)
    
                        auc_mean, auc_std = get_stats(best_aucs)
                        aupr_mean, aupr_std = get_stats(best_auprs)
                        acc_mean, acc_std = get_stats(best_accs)
                        kappa_mean, kappa_std = get_stats(best_kappas)
                        bacc_mean, bacc_std = get_stats(best_baccs)
                        f1_mean, f1_std = get_stats(best_f1s)

                        write_summary_total(args, config_str, [auc_mean, auc_std, aupr_mean, aupr_std, acc_mean, acc_std, kappa_mean, kappa_std, bacc_mean, bacc_std, f1_mean, f1_std])
    
    

def main(args, train_df, test_df, gene_exp_dict, repeat = 0, fold = 0):

    if args.embedder == 'CausalDDS':
        from models import CausalDDS_ModelTrainer
        embedder = CausalDDS_ModelTrainer(args, train_df, test_df, gene_exp_dict, repeat, fold)

    best_auc, best_aupr, best_acc, best_kappa, best_bacc, best_f1 = embedder.train()

    return [best_auc, best_aupr, best_acc, best_kappa, best_bacc, best_f1], embedder.config_str, embedder.best_config_auc


if __name__ == "__main__":
    experiment()


