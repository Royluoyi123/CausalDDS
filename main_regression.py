import pandas as pd
import torch
import random
import numpy as np
import os
import argument
import time
from utils import get_stats, write_summary_reg, write_summary_total_reg

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
        for args.lr in [1e-3,1e-4]:
            for args.weight_decay in [1e-5]:
                for args.dropout in [0,0.1,0.2,0.3,0.4,0.5]: 
                    best_mses, best_rmses, best_r2s, best_pears, best_maes, best_spears = [], [], [], [], [], []
                    for repeat in range(1, args.repeat + 1):
                        for fold in range(1, 6):

                            train_set = torch.load("DDS/ddsdata/"+str(args.dataset)+"/processed/regression/"+args.setting+"/train_"+str(fold)+".pt")
                            test_set = torch.load("DDS/ddsdata/"+str(args.dataset)+"/processed/regression/"+args.setting+"/test_"+str(fold)+".pt")
            
                            print("Dataset Loaded! ({:.4f} sec)".format(time.time() - start))
                            stats, config_str, _,  = main(args, train_set, test_set, gene_exp_dict, repeat = repeat, fold = fold)
        
                            # get Stats
                            best_mses.append(stats[0])
                            best_rmses.append(stats[1])
                            best_pears.append(stats[2])
                            best_r2s.append(stats[3])
                            best_maes.append(stats[4])
                            best_spears.append(stats[5])

                            write_summary_reg(args, config_str, stats)

                            print(fold)
    
                        mse_mean, mse_std = get_stats(best_mses)
                        rmse_mean, rmse_std = get_stats(best_rmses)
                        r2_mean, r2_std = get_stats(best_r2s)
                        pear_mean, pear_std = get_stats(best_pears)
                        mae_mean, mae_std = get_stats(best_maes)
                        spear_mean, spear_std = get_stats(best_spears)

                        write_summary_total_reg(args, config_str, [mse_mean, mse_std, rmse_mean, rmse_std, r2_mean, r2_std, pear_mean, pear_std, mae_mean, mae_std, spear_mean, spear_std])
    
    

def main(args, train_df, test_df, gene_exp_dict, repeat = 0, fold = 0):

    if args.embedder == 'CMRL':
        from models import CMRL_ModelTrainer_reg
        embedder = CMRL_ModelTrainer_reg(args, train_df, test_df, gene_exp_dict, repeat, fold)

    best_mse, best_rmse, best_r2, best_pear, best_mae, best_spear = embedder.train()

    return [best_mse, best_rmse, best_r2, best_pear, best_mae, best_spear], embedder.config_str, embedder.best_config_auc


if __name__ == "__main__":
    experiment()


