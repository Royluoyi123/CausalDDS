import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random
from sklearn.model_selection import KFold

def generate_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)

def scaffold_kfold_split(data_path, k=5, seed=42, binarize_threshold=10):
    smiles_df = pd.read_csv("DDS/ddsdata/DrugCombDB640/drug_smiles.csv",header=0)
    smiles_dict = dict(zip(smiles_df['Drug_name'],smiles_df['Smiles']))
    #smiles_dict = dict(zip(smiles_df['pubchemid'],smiles_df['isosmiles']))
    df = pd.read_csv(data_path)
    assert df.shape[1] >= 4, "数据至少应包含4列：drug1, drug2, cell_line, label"

    # 标签二值化（>= threshold 为 1，否则为 0）
    #df.iloc[:, 3] = df.iloc[:, 3].apply(lambda x: 1 if x >= binarize_threshold else 0)

    # 提取 scaffold 对组合
    scaffold_groups = defaultdict(list)
    for idx, (d1, d2) in enumerate(zip(df.iloc[:, 0], df.iloc[:, 1])):
        scaf1 = generate_scaffold(smiles_dict[d1])
        scaf2 = generate_scaffold(smiles_dict[d2])
        if scaf1 is None or scaf2 is None:
            continue
        key = tuple(sorted([scaf1, scaf2]))
        scaffold_groups[key].append(idx)

    # 将 scaffold group 列表打乱并切分为 K 折
    scaffold_group_list = list(scaffold_groups.values())
    random.seed(seed)
    random.shuffle(scaffold_group_list)

    folds = [[] for _ in range(k)]
    for i, group in enumerate(scaffold_group_list):
        folds[i % k].extend(group)

    # 输出每一折的数据集
    for i in range(k):
        test_idx = folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in folds[j]]

        #columns=['drug_a','drug_b','cell_line','synergy']


        df.iloc[train_idx].to_csv(f"DDS/ddsdata/DrugCombDB640/raw/regression/scaffold_split/train_{i+1}.csv", index=False)
        df.iloc[test_idx].to_csv(f"DDS/ddsdata/DrugCombDB640/raw/regression/scaffold_split/test_{i+1}.csv", index=False)

        # 打印当前折的统计
        train_1 = (df.iloc[train_idx].iloc[:, 3] == 1).sum()
        test_1 = (df.iloc[test_idx].iloc[:, 3] == 1).sum()
        print(f"[Fold {i+1}] Train size: {len(train_idx)} (Label=1: {train_1}), Test size: {len(test_idx)} (Label=1: {test_1})")

    print("1")

# 示例调用
if __name__ == "__main__":
    scaffold_kfold_split("DDS/ddsdata/DrugCombDB640/drug_synergy.csv", k=5, binarize_threshold=10)
