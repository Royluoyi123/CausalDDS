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
from utils import create_batch_mask, get_roc_score
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
        self.explain_loader = DataLoader(self.train_dataset, batch_size = 1)

        self.is_early_stop = False

        self.best_train_auc = -1.0
        self.best_test_auc = -1.0


        self.test_auc = -1.0
        self.test_aupr = -1.0
        self.test_acc = -1.0
        self.test_kappa = -1.0
       

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

        self.train_auc_score, train_aupr_score, train_acc_score, self.train_kappa_score , self.train_bacc_score, self.train_f1_score= get_roc_score(train_outputs, train_labels)


        train_res = PrettyTable()
        train_res.field_names = ["epoch", "train_AUC","train_AUPR","train_ACC","train_KAPPA","train_bacc","train_f1"]
        train_res.add_row(
            [epoch,self.train_auc_score,train_aupr_score,train_acc_score,self.train_kappa_score,self.train_bacc_score,self.train_f1_score]
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

        

        test_auc_score, test_aupr_score, test_acc_score, test_kappa_score, test_bacc_score, test_f1_score = get_roc_score(test_outputs, test_labels)

        #self.writer.add_scalar("test/test_auc_score", test_auc_score, epoch)
        #self.writer.add_scalar("test/test_aupr_score", test_aupr_score, epoch)
        #self.writer.add_scalar("test/test_acc_score", test_acc_score, epoch)
        #self.writer.add_scalar("test/test_kappa_score", test_kappa_score, epoch)
        test_res = PrettyTable()
        test_res.field_names = ["epoch", "test_AUC","test_AUPR","test_ACC","test_KAPPA","test_BACC","test_F1"]
        test_res.add_row(
            [epoch,test_auc_score,test_aupr_score,test_acc_score,test_kappa_score,test_bacc_score,test_f1_score]
            )
        print(test_res)

        # Save ROC score
        if test_auc_score > self.best_test_auc :

            self.patience = 0
            
            # Save train score
            self.best_test_auc = test_auc_score

            # Save test score
            self.test_auc = test_auc_score
            self.test_aupr = test_aupr_score
            self.test_acc = test_acc_score
            self.test_kappa = test_kappa_score
            self.test_bacc = test_bacc_score
            self.test_f1 = test_f1_score
            self.test_outputs = test_outputs
            self.test_labels = test_labels

            # Save epoch
            self.best_auc_epoch = epoch

            if self.args.save_checkpoints == True:
                
                #checkpoint = {'mol_id': np.hstack(mol_indexs), 'output': np.hstack(outputs)}                
                check_dir =  'warm_start_bestepoch_visual_DB.pth'
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
        self.best_config_auc = "[Best Train AUC Epoch: {}] Best epoch Test AUC: {:.4f} / AUPR: {:.4f}  Test ACC: {:.4f} / KAPPA: {:.4f} ".format(self.best_auc_epoch, self.test_auc, self.test_aupr, self.test_acc, self.test_kappa)

        #print(self.eval_config)
        print(self.best_config_auc)
    
