from calendar import c
import numpy as np
import pandas as pd
import torch
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os

from argument import config2string
from utils import create_batch_mask, get_mse_score
from data import Dataclass
from torch_geometric.data import DataLoader

import datetime
from prettytable import PrettyTable

class embedder:

    def __init__(self, args, train_df, test_df, gene_exp_dict, repeat, fold):
        self.args = args

        d = datetime.datetime.now()
        date = d.strftime("%x")[-2:] + d.strftime("%x")[0:2] + d.strftime("%x")[3:5]

        self.config_str = "{}_".format(date) + config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        
        #if args.writer:
            #self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))
        #else:
            #self.writer = SummaryWriter(log_dir="runs_/{}".format(self.config_str))

        # Model Checkpoint Path
        CHECKPOINT_PATH = "model_checkpoints/{}/".format(args.embedder)
        self.check_dir = CHECKPOINT_PATH + self.config_str + ".pt"

        # Select GPU device
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self.device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self.device)

        self.train_dataset = Dataclass(train_df, gene_exp_dict)
        self.test_dataset = Dataclass(test_df, gene_exp_dict)
        self.explain_dataset = Dataclass(train_df, gene_exp_dict)
        
        self.train_loader = DataLoader(self.train_dataset, batch_size = args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size = args.batch_size)
        self.explain_loader = DataLoader(self.test_dataset, batch_size = 32)

        self.is_early_stop = False

        self.best_train_mse = 1000000
        self.best_test_mse = 1000000


        self.test_mse = -1.0
        self.test_rmse = -1.0
        self.test_r2 = -1.0
        self.test_spear = -1.0
        self.test_mae = -1.0
        self.test_pear = -1.0
       

    def evaluate(self, epoch, final = False):
        
        train_outputs, train_labels = [], []
        test_outputs, test_labels = [], []

        mol_indexs, outputs = [], []

        for bc, samples in enumerate(self.train_loader):

            masks = create_batch_mask(samples)
            output,_,_ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], test = True, causal=True)
            
            if self.args.save_checkpoints:
                mol_indexs.append(np.asarray(samples[0]["idx"].detach().cpu()))
                outputs.append(np.asarray(output.reshape(-1).detach().cpu()))

            train_outputs.append(output.reshape(-1).detach().cpu().numpy())
            train_labels.append(samples[3].reshape(-1).detach().cpu().numpy())
        
        train_outputs = np.hstack(train_outputs)
        train_labels = np.hstack(train_labels)

        self.train_mse_score, train_rmse_score, train_pear_score, self.train_r2_score , self.train_mae_score, self.train_spear_score= get_mse_score(train_outputs, train_labels)


        train_res = PrettyTable()
        train_res.field_names = ["epoch", "train_MSE","train_RMSE","train_PEAR","train_R2","train_MAE","train_SPEAR"]
        train_res.add_row(
            [epoch,self.train_mse_score,train_rmse_score,train_pear_score,self.train_r2_score,self.train_mae_score,self.train_spear_score]
            )
        print(train_res)
        #print("train/train_auc_score", self.train_auc_score, epoch)
        #print("train/train_aupr_score", train_aupr_score, epoch)
        #print("train/train_acc_score", train_acc_score, epoch)
        #print("train/train_kappa_score", self.train_kappa_score, epoch)


        for bc, samples in enumerate(self.test_loader):
                
            masks = create_batch_mask(samples)
            output,_,_ = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], test = True,causal = True)
            
            if self.args.save_checkpoints:
                mol_indexs.append(np.asarray(samples[0]["idx"].detach().cpu()))
                outputs.append(np.asarray(output.reshape(-1).detach().cpu()))

            test_outputs.append(output.reshape(-1).detach().cpu().numpy())
            test_labels.append(samples[3].reshape(-1).detach().cpu().numpy())

        test_outputs = np.hstack(test_outputs)
        test_labels = np.hstack(test_labels)

        

        test_mse_score, test_rmse_score, test_pear_score, test_r2_score, test_mae_score, test_spear_score = get_mse_score(test_outputs, test_labels)

        #self.writer.add_scalar("test/test_auc_score", test_auc_score, epoch)
        #self.writer.add_scalar("test/test_aupr_score", test_aupr_score, epoch)
        #self.writer.add_scalar("test/test_acc_score", test_acc_score, epoch)
        #self.writer.add_scalar("test/test_kappa_score", test_kappa_score, epoch)
        test_res = PrettyTable()
        test_res.field_names = ["epoch", "test_MSE","test_RMSE","test_PEAR","test_R2","test_MAE","test_SPEAR"]
        test_res.add_row(
            [epoch,test_mse_score,test_rmse_score,test_pear_score,test_r2_score,test_mae_score,test_spear_score]
            )
        print(test_res)

        # Save ROC score
        if test_mse_score < self.best_test_mse :

            self.patience = 0
            
            # Save train score
            self.best_test_mse = test_mse_score

            # Save test score
            self.test_mse = test_mse_score
            self.test_rmse = test_rmse_score
            self.test_pear = test_pear_score
            self.test_r2 = test_r2_score
            self.test_mae = test_mae_score
            self.test_spear = test_spear_score
            self.test_outputs = test_outputs
            self.test_labels = test_labels

            # Save epoch
            self.best_mse_epoch = epoch

            if self.args.save_checkpoints == True:
                
                #checkpoint = {'mol_id': np.hstack(mol_indexs), 'output': np.hstack(outputs)}                
                check_dir =  'warm_start_bestepoch.pth'
                torch.save(self.model.state_dict(), check_dir)
        
        else:
            self.patience += 1
        
        # Save f1 score
        #if self.train_acc_score > self.best_val_acc :

            #self.best_val_f1 = val_f1_score
            #self.best_val_acc = self.val_acc_score
            
            #self.best_test_f1 = test_f1_score
            #self.best_test_acc = test_acc_score
            
            # Save epoch
            #self.best_f1_epoch = epoch

            #if self.args.save_checkpoints == True:
                
                #checkpoint = {'mol_id': np.hstack(mol_indexs), 'output': np.hstack(outputs)}                
                #check_dir =  self.check_dir[:-3] + '_bestepoch.pt'
                #torch.save(checkpoint, check_dir)
        #self.eval_config = "[Epoch: {} ({:.4f} sec)] Train AUC: {:.4f} / AUPR: {:.4f} / ACC: {:.4f} / KAPPA: {:.4f} || Test AUC: {:.4f} / AUPR: {:.4f} / ACC: {:.4f} / KAPPA: {:.4f} ".format(epoch, self.epoch_time, self.train_auc_score, train_aupr_score, train_acc_score, self.train_kappa_score, test_auc_score, test_aupr_score, test_acc_score, test_kappa_score)
        self.best_config_auc = "[Best Train MSE Epoch: {}] Best epoch Test MSE: {:.4f} / RMSE: {:.4f}  Test PEAR: {:.4f} / R2: {:.4f} ".format(self.best_mse_epoch, self.test_mse, self.test_rmse, self.test_pear, self.test_r2)

        #print(self.eval_config)
        print(self.best_config_auc)
    
