from concurrent.futures import thread
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
seed=3407
cv_mode = 2 #1 随机划分 2 细胞系冷启动 3 药物对冷启动 4 药物
synergy_cv=pd.read_csv('DDS/ddsdata/ALMANAC640/drug_synergy.csv',header=0)
synergy_cv.iloc[:, :2] = synergy_cv.iloc[:, :2].astype(str)
synergy_cv=synergy_cv.to_numpy()
threshold = 10
for row in synergy_cv:
        row[3] = 1 if row[3] >= threshold else 0
if cv_mode == 1:
    cv_data = synergy_cv
elif cv_mode == 2:
    cv_data = np.unique(synergy_cv[:, 2])  # cline_level
elif cv_mode == 3:
    drug = synergy_cv[:, [0, 1]].astype(str)  # 强制转换为字符串类型
    cv_data = np.unique(drug, axis=0)
    #cv_data = np.unique(synergy_cv[:, [0, 1]], axis=0) # drug pairs_level
elif cv_mode == 4:
    all_drugs = np.unique(np.concatenate([synergy_cv[:, 0].astype(str),
                                          synergy_cv[:, 1].astype(str)]))
    cv_data = all_drugs
final_metric = np.zeros(4)
fold_num = 1
kf = KFold(n_splits=5, shuffle=True, random_state=seed)
for train_index, validation_index in kf.split(cv_data):
    # ---construct train_set+validation_set
    if cv_mode == 1:  # normal_level
        synergy_train, synergy_validation = cv_data[train_index], cv_data[validation_index]
    elif cv_mode == 2:  # cell line_level
        train_name, test_name = cv_data[train_index], cv_data[validation_index]
        synergy_train = np.array([i for i in synergy_cv if i[2] in train_name])
        synergy_validation = np.array([i for i in synergy_cv if i[2] in test_name])
    elif cv_mode == 3:  # drug pairs_level
        pair_train, pair_validation = cv_data[train_index], cv_data[validation_index]
        synergy_train = np.array(
            [j for i in pair_train for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
        synergy_validation = np.array(
            [j for i in pair_validation for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
    elif cv_mode == 4:
        train_drugs, test_drugs = cv_data[train_index], cv_data[validation_index]
        mask_train = np.isin(synergy_cv[:, 0], train_drugs) & np.isin(synergy_cv[:, 1], train_drugs)
        synergy_train = synergy_cv[mask_train]


        mask_val = np.isin(synergy_cv[:, 0], test_drugs) | np.isin(synergy_cv[:, 1], test_drugs)
        synergy_validation = synergy_cv[mask_val]

    columns=['drug_a','drug_b','cell_line','synergy']
    df=pd.DataFrame(synergy_train,columns=columns)
    df.to_csv('DDS/ddsdata/ALMANAC640/raw/classification10/cell_line/train_'+str(fold_num)+'.csv', index=False)
    df.to_csv(f'DDS/ddsdata/ALMANAC640/raw/classification10/cell_line/train_{fold_num}.txt',
              index=False, header=False, sep='\t')

    df=pd.DataFrame(synergy_validation,columns=columns)
    df.to_csv('DDS/ddsdata/ALMANAC640/raw/classification10/cell_line/test_'+str(fold_num)+'.csv', index=False)
    df.to_csv(f'DDS/ddsdata/ALMANAC640/raw/classification10/cell_line/test_{fold_num}.txt',
              index=False, header=False, sep='\t')
    
    print(fold_num)
    fold_num +=1