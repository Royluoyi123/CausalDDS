import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

import numpy as np

from torch_geometric.nn import Set2Set, global_mean_pool

from embedder import embedder
from layers import GatherModel, GINE
from utils import create_batch_mask, create_interv_mask

from torch_scatter import scatter_mean, scatter_add, scatter_std

import random
import time

class CMRL_ModelTrainer(embedder):
    
    def __init__(self, args, train_df, test_df, gene_exp_dict, repeat, fold):
        embedder.__init__(self, args, train_df, test_df, gene_exp_dict, repeat, fold)

        self.num_classes = 2

        self.model = CMRL(args, device = self.device, num_step_message_passing = self.args.message_passing, num_classes = self.num_classes, intervention = self.args.intervention, conditional = self.args.conditional).to(self.device)
        self.optimizer = optim.Adam(params = self.model.parameters(), lr = self.args.lr, weight_decay = self.args.weight_decay)
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=self.args.patience, mode='max', verbose=True) #学习率调节器
        
    def train(self):
        
        loss_function_BCE = nn.BCEWithLogitsLoss(reduction='mean')
        self.patience = 0
        
        for epoch in range(1, self.args.epochs + 1):
            self.model.train()
            self.train_loss = 0
            self.loss_pos = 0
            self.loss_neg = 0
            self.loss_inv = 0
            self.importance_A = 0
            self.importance_B = 0

            start = time.time()

            for bc, samples in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                masks = create_batch_mask(samples) #创建稀疏矩阵存储

                #samples[0] 药物1, samples[1] 药物2, samples[2] 细胞系, samples[3] label

                pred = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)])
                loss = loss_function_BCE(pred, samples[3].reshape(-1, 1).to(self.device).float()).mean() # Supervised Loss 没有任何对图操作的

                if self.args.symmetric:
                    if epoch % 2 == 0:
                        # Causal Loss
                        pos, neg, rand = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], causal = True)
                    else:
                        pos, neg, rand = self.model([samples[1].to(self.device), samples[0].to(self.device), masks[1].to(self.device), masks[0].to(self.device), samples[2].to(self.device)], causal = True)
                
                else:
                    pos, neg, rand = self.model([samples[0].to(self.device), samples[1].to(self.device), masks[0].to(self.device), masks[1].to(self.device), samples[2].to(self.device)], causal = True)
                
                loss_pos = loss_function_BCE(pos, samples[3].reshape(-1, 1).to(self.device).float()) #有效部分的loss
                
                random_label = torch.ones_like(neg, dtype=torch.float).to(self.device) / self.num_classes
                loss_neg = F.kl_div(neg, random_label, reduction = 'batchmean')
                
                loss = loss + loss_pos + self.args.lam1 * loss_neg

                #print(loss,loss_pos,self.args.lam1 * loss_neg)
                
                # Intervention Loss
                if self.args.intervention:
                    loss_inv = self.args.lam2 * loss_function_BCE(rand, samples[3].reshape(-1, 1).to(self.device).float())
                    loss += loss_inv
                    self.loss_inv += loss_inv
                
                loss.backward()
                self.optimizer.step()
                self.train_loss += loss
                
                self.loss_pos += loss_pos
                self.loss_neg += loss_neg
                self.importance_A += torch.sigmoid(self.model.importance_A).mean() 
                self.importance_B += torch.sigmoid(self.model.importance_A).mean()
            
            self.epoch_time = time.time() - start

            self.model.eval()
            self.evaluate(epoch)

            self.scheduler.step(self.train_auc_score)
            
            # Write Statistics
            #self.writer.add_scalar("loss/positive", self.loss_pos/bc, epoch)
            #self.writer.add_scalar("loss/negative", self.loss_neg/bc, epoch)
            #if self.args.intervention:
                #self.writer.add_scalar("loss/intervention", self.loss_inv/bc, epoch)
            #self.writer.add_scalar("stats/importance drug A", self.importance_A/bc, epoch)
            #self.writer.add_scalar("stats/importance drug B", self.importance_A/bc, epoch)

            #print("loss/positive", self.loss_pos/bc, epoch)
            #print("loss/negative", self.loss_neg/bc, epoch)
            #if self.args.intervention:
                #print("loss/intervention", self.loss_inv/bc, epoch)
            #print("stats/importance drug A", self.importance_A/bc, epoch)
            #print("stats/importance drug B", self.importance_B/bc, epoch)
            print(self.patience)
            

            # Early stopping
            if self.patience > int(self.args.es / self.args.eval_freq):
                        break
   
        return self.test_auc, self.test_aupr, self.test_acc, self.test_kappa, self.test_bacc


class CMRL(nn.Module):
    """
    This the main class for CIGIN model
    """

    def __init__(self,
                args,
                device,
                node_input_dim=133,
                edge_input_dim=14,
                node_hidden_dim=300,
                edge_hidden_dim=300,
                num_step_message_passing=3,
                num_step_set2_set=2,
                num_layer_set2set=1,
                num_classes = 2,
                intervention = False,
                conditional = False
                ):
        super(CMRL, self).__init__()

        self.device = device
        self.node_input_dim = node_input_dim #133
        self.node_hidden_dim = node_hidden_dim #300
        self.edge_input_dim = edge_input_dim #14
        self.edge_hidden_dim = edge_hidden_dim #300
        self.num_step_message_passing = num_step_message_passing #3
        self.intervention = intervention #True
        self.conditional = conditional #True
        self.drop_out = args.dropout

        self.gather = GINE(self.node_input_dim, self.edge_input_dim, 
                            self.node_hidden_dim, self.num_step_message_passing,
                            )
        
        self.compressor = nn.Sequential(
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.node_hidden_dim, 1)
            )

        #self.predictor = nn.Linear(8 * self.node_hidden_dim + 640, 1)

        self.predictor = nn.Sequential(
            nn.Linear(8 * self.node_hidden_dim + 640, 4* self.node_hidden_dim),
            nn.BatchNorm1d(4* self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(4 * self.node_hidden_dim, 2 * self.node_hidden_dim),
            nn.BatchNorm1d(2 * self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.node_hidden_dim, 1)
            )

        #self.neg_predictor = nn.Linear(4 * self.node_hidden_dim + 640, num_classes)

        self.neg_predictor = nn.Sequential(
            nn.Linear(4 * self.node_hidden_dim + 640, 2* self.node_hidden_dim),
            nn.BatchNorm1d(2* self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(2 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.node_hidden_dim, 2)
            )

        #self.rand_predictor = nn.Linear(12 * self.node_hidden_dim + 640, 1)

        self.rand_predictor = nn.Sequential(
            nn.Linear(12 * self.node_hidden_dim + 640, 6* self.node_hidden_dim),
            nn.BatchNorm1d(6* self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(6 * self.node_hidden_dim, self.node_hidden_dim),
            nn.BatchNorm1d(self.node_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.drop_out),
            nn.Linear(self.node_hidden_dim, 1)
            )

        self.num_step_set2set = num_step_set2_set #2
        self.num_layer_set2set = num_layer_set2set #1
        self.set2set_pos_drugA = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set) #讲图的节点集合转换成一个表示：得到图的表征？
        self.set2set_pos_drugB = Set2Set(2 * node_hidden_dim, self.num_step_set2set, self.num_layer_set2set)

        self.init_model()
    
    
    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    

    def compress(self, drugA_features):
        
        p = self.compressor(drugA_features) #公式4
        temperature = 1.0
        bias = 0.0 + 0.0001  # If bias is 0, we run into problems
        eps = (bias - (1 - bias)) * torch.rand(p.size()) + (1 - bias) #u
        gate_inputs = torch.log(eps) - torch.log(1 - eps) #公式6后半部分
        gate_inputs = gate_inputs.to(self.device)
        gate_inputs = (gate_inputs + p) / temperature
        gate_inputs = torch.sigmoid(gate_inputs).squeeze()

        return gate_inputs, p
    

    def interaction(self, drugA_features, drugB_features):

        # Do normalization
        normalized_drugA_features = F.normalize(drugA_features, dim = 1) #节点个数X300
        normalized_drugB_features = F.normalize(drugB_features, dim = 1)

        # Interaction phase 这是不是在计算余弦相似度要考究
        len_map = torch.sparse.mm(self.drugA_len.t(), self.drugB_len) 

        interaction_map = torch.mm(normalized_drugA_features, normalized_drugB_features.t()) 
        interaction_map = interaction_map * len_map.to_dense()
        
        #公式3
        drugB_prime = torch.mm(interaction_map.t(), normalized_drugA_features)
        drugA_prime = torch.mm(interaction_map, normalized_drugB_features)

        # Prediction phase 拼接过后的特征
        drugA_features = torch.cat((normalized_drugA_features, drugA_prime), dim=1) #节点个数X600
        drugB_features = torch.cat((normalized_drugB_features, drugB_prime), dim=1)

        return drugA_features, drugB_features
    

    def forward(self, data, causal = False, test = False):

        drugA = data[0] #pygeo 数据
        drugB = data[1]
        self.drugA_len = data[2]
        self.drugB_len = data[3]
        cell_features = data[4]
        # node embeddings after interaction phase
        _drugA_features = self.gather(drugA) #公式2 节点个数X300 GIN得到的分子表征
        _drugB_features = self.gather(drugB) #公式3 节点个数,300

        drugA_features, drugB_features = self.interaction(_drugA_features, _drugB_features) #公式3 前一个是药物1 后一个是药物2

        if (test == True) or (causal == False):

            drugA_features_s2s = self.set2set_pos_drugA(drugA_features, drugA.batch) #512 1200 molecular level representation 4.3最后
            drugB_features_s2s = self.set2set_pos_drugB(drugB_features, drugB.batch) #512 1200

            combine_feature = torch.cat((drugA_features_s2s, drugB_features_s2s, cell_features), 1)
            predictions = self.predictor(combine_feature)


            return predictions
            

        else:

            lambda_pos_A, self.importance_A = self.compress(drugA_features)
            lambda_pos_A = lambda_pos_A.reshape(-1, 1)
            lambda_neg_A = 1 - lambda_pos_A

            # Inject Noise for Drug A
            static_drugA_feature = drugA_features.clone().detach()

            node_drugA_feature_mean = scatter_mean(static_drugA_feature, drugA.batch, dim = 0)[drugA.batch] #14576,600
            node_drugA_feature_std = scatter_std(static_drugA_feature, drugA.batch, dim = 0)[drugA.batch] #14576,600

            noisy_drugA_node_feature_mean = lambda_pos_A * drugA_features + lambda_neg_A * node_drugA_feature_mean #公式五
            noisy_drugA_node_feature_std = lambda_neg_A * node_drugA_feature_std 

            pos_drugA = noisy_drugA_node_feature_mean + torch.rand_like(noisy_drugA_node_feature_mean) * noisy_drugA_node_feature_std
            neg_drugA = lambda_neg_A * drugA_features

            pos_drugA_s2s = self.set2set_pos_drugA(pos_drugA, drugA.batch) #256,1200
            neg_drugA = global_mean_pool(neg_drugA, drugA.batch) #256,600

            # Inject Noise for Drug B

            lambda_pos_B, self.importance_B = self.compress(drugB_features)
            lambda_pos_B = lambda_pos_B.reshape(-1, 1)
            lambda_neg_B = 1 - lambda_pos_B

            static_drugB_feature = drugB_features.clone().detach()

            node_drugB_feature_mean = scatter_mean(static_drugB_feature, drugB.batch, dim = 0)[drugB.batch] #14576,600
            node_drugB_feature_std = scatter_std(static_drugB_feature, drugB.batch, dim = 0)[drugB.batch] #14576,600

            noisy_drugB_node_feature_mean = lambda_pos_B * drugB_features + lambda_neg_B * node_drugB_feature_mean #公式五
            noisy_drugB_node_feature_std = lambda_neg_B * node_drugB_feature_std 

            pos_drugB = noisy_drugB_node_feature_mean + torch.rand_like(noisy_drugB_node_feature_mean) * noisy_drugB_node_feature_std
            neg_drugB = lambda_neg_B * drugB_features

            pos_drugB_s2s = self.set2set_pos_drugB(pos_drugB, drugB.batch)
            neg_drugB = global_mean_pool(neg_drugB, drugB.batch)




            #drugB_features_s2s = self.set2set_drugB(drugB_features, drugB.batch)


            pos_drugA_drugB_cell = torch.cat((pos_drugA_s2s, pos_drugB_s2s, cell_features), 1) #药物A的关键特征加上药物B的关键特征

            neg_drugA_drugB_cell = torch.cat((neg_drugA, neg_drugB, cell_features), 1) #药物A的非关键特征加上药物B的非关键特征

            pos_predictions = self.predictor(pos_drugA_drugB_cell)
            neg_predictions = self.neg_predictor(neg_drugA_drugB_cell)

            if self.intervention == True:

                num = pos_drugA_s2s.shape[0] #256
                l = [i for i in range(num)]
                random.shuffle(l)
                random_idx = torch.tensor(l)

                if self.conditional == True:

                    # Create Intervention Interaction Map
                    rand_batch_drugA = random_idx[drugA.batch]
                    self.drugA_len = create_interv_mask(rand_batch_drugA).to(self.device)

                    rand_batch_drugB = random_idx[drugB.batch]
                    self.drugB_len = create_interv_mask(rand_batch_drugB).to(self.device)

                    drugA_features, drugB_features = self.interaction(_drugA_features, _drugB_features)

                    lambda_pos_drugA, self.importance_drugA = self.compress(drugA_features)
                    lambda_pos_drugA = lambda_pos_drugA.reshape(-1, 1)
                    lambda_neg_drugA = 1 - lambda_pos_drugA

                    rand_drugA = lambda_neg_drugA * drugA_features

                    rand_drugA = global_mean_pool(rand_drugA, drugA.batch)

                    lambda_pos_drugB, self.importance_drugB = self.compress(drugB_features)
                    lambda_pos_drugB = lambda_pos_drugB.reshape(-1, 1)
                    lambda_neg_drugB = 1 - lambda_pos_drugB

                    rand_drugB = lambda_neg_drugB * drugB_features

                    rand_drugB = global_mean_pool(rand_drugB, drugB.batch)

                    rand_drugA_drugB = torch.cat((pos_drugA_s2s, pos_drugB_s2s, rand_drugA[random_idx], rand_drugB[random_idx], cell_features), 1)





                    random_predictions = self.rand_predictor(rand_drugA_drugB)
                
                else:
                    #rand_drugA_drugB = torch.cat((pos_drugA_s2s, drugB_features_s2s, neg_drugA[random_idx]), 1)
                    rand_drugA_drugB = torch.cat((pos_drugA_s2s, pos_drugB_s2s, rand_drugA[random_idx], rand_drugB[random_idx], cell_features), 1)
                    random_predictions = self.rand_predictor(rand_drugA_drugB)

                return pos_predictions, F.log_softmax(neg_predictions, dim = -1), random_predictions
        
            return pos_predictions, F.log_softmax(neg_predictions, dim = -1), None