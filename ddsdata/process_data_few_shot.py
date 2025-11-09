import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 参数设置
seed = 3407
n_repeat = 5
train_ratios = [0.1,0.2,0.3,0.4]
cv_mode = 1  # 1: 随机划分，2: 细胞系冷启动，3: 药物对冷启动

# 读取原始数据
synergy_cv = pd.read_csv('DDS/ddsdata/ALMANAC640/drug_synergy.csv', header=0)
synergy_cv.iloc[:, :2] = synergy_cv.iloc[:, :2].astype(str)  # 强制 drug_a, drug_b 为字符串
synergy_cv = synergy_cv.to_numpy()

threshold = 10
for row in synergy_cv:
        row[3] = 1 if row[3] >= threshold else 0

# 根据模式定义划分对象
if cv_mode == 1:
    cv_data = synergy_cv
elif cv_mode == 2:
    cv_data = np.unique(synergy_cv[:, 2])  # cell line 冷启动
else:
    drug_pairs = synergy_cv[:, [0, 1]].astype(str)
    cv_data = np.unique(drug_pairs, axis=0)  # drug pair 冷启动

# 主循环：比例 × 重复次数
for train_ratio in train_ratios:
    for repeat in range(n_repeat):
        np.random.seed(seed + repeat)  # 每次不同 seed

        if cv_mode == 1:
            train_data, test_data = train_test_split(cv_data, train_size=train_ratio, random_state=seed + repeat, shuffle=True)

        elif cv_mode == 2:
            train_clines, test_clines = train_test_split(cv_data, train_size=train_ratio, random_state=seed + repeat, shuffle=True)
            train_data = np.array([i for i in synergy_cv if i[2] in train_clines])
            test_data = np.array([i for i in synergy_cv if i[2] in test_clines])

        else:  # drug pair 冷启动
            train_pairs, test_pairs = train_test_split(cv_data, train_size=train_ratio, random_state=seed + repeat, shuffle=True)
            train_data = np.array([j for i in train_pairs for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])
            test_data = np.array([j for i in test_pairs for j in synergy_cv if (i[0] == j[0]) and (i[1] == j[1])])

        # 保存文件
        columns = ['drug_a', 'drug_b', 'cell_line', 'synergy']
        pd.DataFrame(train_data, columns=columns).to_csv(
            f'DDS/ddsdata/ALMANAC640/raw/classification10/fewshot{int(train_ratio*100)}/train_{repeat+1}.csv',
            index=False
        )
        pd.DataFrame(test_data, columns=columns).to_csv(
            f'DDS/ddsdata/ALMANAC640/raw/classification10/fewshot{int(train_ratio*100)}/test_{repeat+1}.csv',
            index=False
        )

        print(f"[Done] ratio={train_ratio}, repeat={repeat+1}, train size={len(train_data)}, test size={len(test_data)}")
